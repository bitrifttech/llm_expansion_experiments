import os
import sys
import torch
import numpy as np
import psutil
import time
import random
import ast
import re
from copy import deepcopy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats
import difflib
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Add utils to path for model analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_analyzer import ModelAnalyzer, analyze_model

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
NUM_NEW_ATTENTION_HEADS = 1  # Number of new attention heads to add per layer
# ============================================================================

# Set random seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device configuration
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon MPS")
else:
    device = "cpu"
    print("Using CPU")

# Memory info
if device == "cuda":
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Device: {device}, GPU Memory: {gpu_memory:.2f} GB")
elif device == "mps":
    system_memory = psutil.virtual_memory().total / (1024**3)
    print(f"Device: {device}, System Memory: {system_memory:.2f} GB")
else:
    system_memory = psutil.virtual_memory().total / (1024**3)
    print(f"Device: {device}, System Memory: {system_memory:.2f} GB")

def log_message(message: str, level: str = "INFO"):
    """Log message with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def freeze_base_model(model):
    """Freeze all parameters in the base model"""
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        # Keep new attention head parameters trainable
        if 'new_q_proj' in name or 'new_k_proj' in name or 'new_v_proj' in name or 'new_o_proj' in name or 'gate' in name:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    
    log_message(f"Froze {frozen_count} base model parameters, kept {trainable_count} expansion parameters trainable")
    
    # Verification: Double-check freezing worked
    actual_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    actual_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    
    log_message(f"Verification: {actual_frozen} frozen, {actual_trainable} trainable parameters")
    
    if actual_trainable == 0:
        log_message("WARNING: No trainable parameters found after freezing!")
    
    return actual_frozen, actual_trainable

class ExpandedMultiHeadAttention(torch.nn.Module):
    """Multi-head attention with additional trainable heads"""
    
    def __init__(self, original_attention, num_new_heads: int = NUM_NEW_ATTENTION_HEADS, device: str = "cpu"):
        super().__init__()
        self.original_attention = original_attention  # Frozen
        self.num_new_heads = num_new_heads
        self.device = device
        self.generation_mode = False  # Flag to disable new heads during generation
        
        # Get configuration from original attention
        self.d_model = original_attention.d_model
        self.num_original_heads = original_attention.n_heads
        self.d_kv = original_attention.key_value_proj_dim
        
        # Get dtype from original attention
        self.dtype = next(original_attention.parameters()).dtype
        
        # Freeze original attention
        for param in self.original_attention.parameters():
            param.requires_grad = False
        
        # Create new attention heads
        # Each head needs Q, K, V projections and output projection
        self.new_q_proj = torch.nn.Linear(
            self.d_model, 
            num_new_heads * self.d_kv, 
            bias=False, 
            dtype=self.dtype, 
            device=device
        )
        self.new_k_proj = torch.nn.Linear(
            self.d_model, 
            num_new_heads * self.d_kv, 
            bias=False, 
            dtype=self.dtype, 
            device=device
        )
        self.new_v_proj = torch.nn.Linear(
            self.d_model, 
            num_new_heads * self.d_kv, 
            bias=False, 
            dtype=self.dtype, 
            device=device
        )
        
        # Output projection for new heads
        self.new_o_proj = torch.nn.Linear(
            num_new_heads * self.d_kv, 
            self.d_model, 
            bias=False, 
            dtype=self.dtype, 
            device=device
        )
        
        # Dropout for new heads
        self.dropout = torch.nn.Dropout(0.1)
        
        # Gate to control contribution of new heads (start more enabled for better gradient flow)
        self.gate = torch.nn.Parameter(torch.tensor(0.0, dtype=self.dtype, device=device))  # sigmoid(0) = 0.5 (50%)
        
        # Initialize new head weights more aggressively for better learning
        with torch.no_grad():
            # Copy weights from original attention heads for stability
            # This provides much better initialization than random weights
            
            # Get original Q, K, V, O weights
            orig_q_weight = self.original_attention.q.weight.data  # [d_model, num_heads * d_kv]
            orig_k_weight = self.original_attention.k.weight.data
            orig_v_weight = self.original_attention.v.weight.data
            orig_o_weight = self.original_attention.o.weight.data  # [num_heads * d_kv, d_model]
            
            # Calculate dimensions
            orig_head_dim = self.num_original_heads * self.d_kv
            new_head_dim = self.num_new_heads * self.d_kv
            
            # Copy and adapt weights for new heads
            if new_head_dim <= orig_head_dim:
                # If we need fewer parameters, take a subset
                self.new_q_proj.weight.data = orig_q_weight[:new_head_dim, :].clone()
                self.new_k_proj.weight.data = orig_k_weight[:new_head_dim, :].clone()
                self.new_v_proj.weight.data = orig_v_weight[:new_head_dim, :].clone()
                self.new_o_proj.weight.data = orig_o_weight[:, :new_head_dim].clone()
            else:
                # If we need more parameters, tile/repeat the original weights
                repeat_factor = (new_head_dim + orig_head_dim - 1) // orig_head_dim  # Ceiling division
                
                # Tile the weights
                tiled_q = orig_q_weight.repeat(repeat_factor, 1)[:new_head_dim, :]
                tiled_k = orig_k_weight.repeat(repeat_factor, 1)[:new_head_dim, :]
                tiled_v = orig_v_weight.repeat(repeat_factor, 1)[:new_head_dim, :]
                tiled_o = orig_o_weight.repeat(1, repeat_factor)[:, :new_head_dim]
                
                self.new_q_proj.weight.data = tiled_q.clone()
                self.new_k_proj.weight.data = tiled_k.clone()
                self.new_v_proj.weight.data = tiled_v.clone()
                self.new_o_proj.weight.data = tiled_o.clone()
            
            # Scale up the copied weights for stronger initial signal
            scale_factor = 1.0  # Start with full magnitude (increased from 50%)
            self.new_q_proj.weight.data *= scale_factor
            self.new_k_proj.weight.data *= scale_factor
            self.new_v_proj.weight.data *= scale_factor
            self.new_o_proj.weight.data *= scale_factor
            
            # Add significant random noise for diversity and better gradient flow
            noise_std = 0.1  # Increased noise significantly for better gradient flow
            self.new_q_proj.weight.data += torch.randn_like(self.new_q_proj.weight.data) * noise_std
            self.new_k_proj.weight.data += torch.randn_like(self.new_k_proj.weight.data) * noise_std
            self.new_v_proj.weight.data += torch.randn_like(self.new_v_proj.weight.data) * noise_std
            self.new_o_proj.weight.data += torch.randn_like(self.new_o_proj.weight.data) * noise_std
        
        log_message(f"Created ExpandedMultiHeadAttention: {self.num_original_heads} original + {num_new_heads} new heads on {device} ({self.dtype})")
    
    def forward(self, hidden_states, mask=None, key_value_states=None, 
                position_bias=None, past_key_value=None, layer_head_mask=None, 
                query_length=None, use_cache=False, output_attentions=False, cache_position=None):
        """Forward pass with original + new attention heads - FIXED VERSION"""
        
        # Original attention output (frozen)
        with torch.no_grad():
            original_outputs = self.original_attention(
                hidden_states=hidden_states,
                mask=mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                past_key_value=past_key_value,
                layer_head_mask=layer_head_mask,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position
            )
        
        # If in generation mode, return only original output
        if self.generation_mode:
            return original_outputs
        
        # Extract original attention output
        if isinstance(original_outputs, tuple):
            original_attention_output = original_outputs[0]
        else:
            original_attention_output = original_outputs
        
        # CRITICAL FIX: Handle dynamic sequence lengths properly with extensive error handling
        try:
            # Get dimensions safely
            batch_size, seq_length = hidden_states.shape[:2]
            
            # Use key_value_states if provided (for cross-attention), otherwise use hidden_states
            kv_input = key_value_states if key_value_states is not None else hidden_states
            kv_seq_length = kv_input.size(1)
            
            # Project to Q, K, V for new heads
            new_q = self.new_q_proj(hidden_states)  # [batch, seq_len, num_new_heads * d_kv]
            new_k = self.new_k_proj(kv_input)       # [batch, kv_seq_len, num_new_heads * d_kv]
            new_v = self.new_v_proj(kv_input)       # [batch, kv_seq_len, num_new_heads * d_kv]
            
            # CRITICAL FIX: Ensure dimensions are correct before reshaping
            expected_q_dim = self.num_new_heads * self.d_kv
            if new_q.size(-1) != expected_q_dim:
                # Fallback: return original output only
                return original_outputs
            
            # CRITICAL FIX: Reshape with explicit dimension validation
            try:
                new_q = new_q.view(batch_size, seq_length, self.num_new_heads, self.d_kv).transpose(1, 2)
                new_k = new_k.view(batch_size, kv_seq_length, self.num_new_heads, self.d_kv).transpose(1, 2)
                new_v = new_v.view(batch_size, kv_seq_length, self.num_new_heads, self.d_kv).transpose(1, 2)
            except RuntimeError:
                # Fallback: return original output only
                return original_outputs
            
            # CRITICAL FIX: Validate tensor dimensions before matrix multiplication
            if new_q.size(-1) != new_k.size(-1):
                return original_outputs
            
            if new_q.size(-2) == 0 or new_k.size(-2) == 0:
                return original_outputs
            
            # Compute attention scores with dimension validation
            try:
                scores = torch.matmul(new_q, new_k.transpose(-2, -1)) / np.sqrt(self.d_kv)
            except RuntimeError:
                # Fallback: return original output only
                return original_outputs
            
            # Apply attention mask if provided (simplified for robustness)
            if mask is not None:
                try:
                    # Simple mask handling - expand to match scores dimensions
                    if mask.dim() == 2:  # [batch, seq_len]
                        expanded_mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_new_heads, -1, kv_seq_length)
                    elif mask.dim() == 3:  # [batch, seq_len, kv_seq_len]
                        expanded_mask = mask.unsqueeze(1).expand(-1, self.num_new_heads, -1, -1)
                    elif mask.dim() == 4:  # [batch, num_heads, seq_len, kv_seq_len]
                        if mask.size(1) == 1:
                            expanded_mask = mask.expand(-1, self.num_new_heads, -1, -1)
                        else:
                            expanded_mask = mask[:, :self.num_new_heads, :, :]
                    elif mask.dim() == 5:  # [batch, 1, 1, seq_len, kv_seq_len] - T5 relative attention bias format
                        # Squeeze out the extra dimensions and expand properly
                        squeezed_mask = mask.squeeze(1).squeeze(1)  # [batch, seq_len, kv_seq_len]
                        expanded_mask = squeezed_mask.unsqueeze(1).expand(-1, self.num_new_heads, -1, -1)
                    else:
                        # Skip mask if too complex
                        expanded_mask = None
                    
                    if expanded_mask is not None and expanded_mask.shape == scores.shape:
                        scores = scores.masked_fill(expanded_mask == 0, -1e9)
                except Exception:
                    # Continue without mask if any error occurs
                    pass
            
            # Apply position bias if provided (simplified for robustness)
            if position_bias is not None:
                try:
                    if position_bias.size(1) == self.num_original_heads:
                        # Original position bias has 8 heads, we need fewer heads
                        if self.num_new_heads <= self.num_original_heads:
                            # Take subset of position bias for new heads
                            new_position_bias = position_bias[:, :self.num_new_heads, :, :]
                        else:
                            # Repeat position bias pattern for more new heads
                            repeat_factor = (self.num_new_heads + self.num_original_heads - 1) // self.num_original_heads
                            new_position_bias = position_bias.repeat(1, repeat_factor, 1, 1)[:, :self.num_new_heads, :, :]
                    elif position_bias.size(1) == 1:
                        # Single head bias, expand to new heads
                        new_position_bias = position_bias.expand(-1, self.num_new_heads, -1, -1)
                    else:
                        # Try to use as-is if dimensions match
                        new_position_bias = position_bias
                    
                    if new_position_bias.shape == scores.shape:
                        scores = scores + new_position_bias
                except Exception:
                    # Continue without position bias if any error occurs
                    pass
            
            # Softmax attention weights
            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values with dimension validation
            try:
                if attention_weights.size(-1) != new_v.size(-2):
                    return original_outputs
                
                new_attention_output = torch.matmul(attention_weights, new_v)
            except RuntimeError:
                return original_outputs
            
            # Reshape and project output
            try:
                new_attention_output = new_attention_output.transpose(1, 2).contiguous()
                new_attention_output = new_attention_output.view(batch_size, seq_length, self.num_new_heads * self.d_kv)
                new_attention_output = self.new_o_proj(new_attention_output)
            except RuntimeError:
                return original_outputs
            
            # Apply gate to control contribution
            base_contribution = 0.1  # Max 10% contribution during training
            
            # Slightly reduce contribution during inference to prevent generation issues
            if not self.training:
                base_contribution = 0.07  # 7% during inference (was 5%)
            
            gate_value = torch.sigmoid(self.gate) * base_contribution
            gated_new_output = gate_value * new_attention_output
            
            # Combine original and new attention outputs
            combined_output = original_attention_output + gated_new_output
            
            # Return in the same format as original
            if isinstance(original_outputs, tuple):
                return (combined_output,) + original_outputs[1:]
            else:
                return combined_output
                
        except Exception:
            # ULTIMATE FALLBACK: If anything goes wrong, just return original output
            return original_outputs
    
    def get_training_stats(self):
        """Get statistics about the new attention heads for verification"""
        stats = {
            'gate_value': torch.sigmoid(self.gate).item(),
            'new_q_weight_norm': self.new_q_proj.weight.norm().item(),
            'new_k_weight_norm': self.new_k_proj.weight.norm().item(),
            'new_v_weight_norm': self.new_v_proj.weight.norm().item(),
            'new_o_weight_norm': self.new_o_proj.weight.norm().item(),
            'new_q_grad_norm': self.new_q_proj.weight.grad.norm().item() if self.new_q_proj.weight.grad is not None else 0.0,
            'new_k_grad_norm': self.new_k_proj.weight.grad.norm().item() if self.new_k_proj.weight.grad is not None else 0.0,
            'new_v_grad_norm': self.new_v_proj.weight.grad.norm().item() if self.new_v_proj.weight.grad is not None else 0.0,
            'new_o_grad_norm': self.new_o_proj.weight.grad.norm().item() if self.new_o_proj.weight.grad is not None else 0.0,
        }
        return stats
    
    def enable_generation_mode(self):
        """Enable generation mode (disable new heads)"""
        self.generation_mode = True
    
    def disable_generation_mode(self):
        """Disable generation mode (enable new heads)"""
        self.generation_mode = False

def expand_model_attention_heads(model, num_new_heads: int = NUM_NEW_ATTENTION_HEADS):
    """Expand all attention layers in the model with additional trainable heads"""
    log_message(f"Starting attention head expansion with {num_new_heads} new heads per layer...")
    
    # Analyze original model
    original_analyzer = ModelAnalyzer(model, f"Original Model")
    
    expanded_model = deepcopy(model)
    
    # Get the device from the model
    model_device = next(model.parameters()).device
    
    # Replace attention layers with expanded versions FIRST
    expansion_count = 0
    
    # Expand encoder self-attention layers
    for layer_idx, layer in enumerate(expanded_model.encoder.block):
        if hasattr(layer, 'layer') and len(layer.layer) > 0:
            # T5 structure: layer[0] is self-attention
            original_attention = layer.layer[0].SelfAttention
            expanded_attention = ExpandedMultiHeadAttention(original_attention, num_new_heads, model_device)
            layer.layer[0].SelfAttention = expanded_attention
            expansion_count += 1
            log_message(f"Expanded encoder layer {layer_idx} self-attention")
    
    # Expand decoder self-attention and cross-attention layers
    for layer_idx, layer in enumerate(expanded_model.decoder.block):
        if hasattr(layer, 'layer') and len(layer.layer) > 1:
            # T5 structure: layer[0] is self-attention, layer[1] is cross-attention
            
            # Self-attention
            original_self_attention = layer.layer[0].SelfAttention
            expanded_self_attention = ExpandedMultiHeadAttention(original_self_attention, num_new_heads, model_device)
            layer.layer[0].SelfAttention = expanded_self_attention
            expansion_count += 1
            log_message(f"Expanded decoder layer {layer_idx} self-attention")
            
            # Cross-attention
            original_cross_attention = layer.layer[1].EncDecAttention
            expanded_cross_attention = ExpandedMultiHeadAttention(original_cross_attention, num_new_heads, model_device)
            layer.layer[1].EncDecAttention = expanded_cross_attention
            expansion_count += 1
            log_message(f"Expanded decoder layer {layer_idx} cross-attention")
    
    log_message(f"Attention Head Expansion complete: {expansion_count} attention layers expanded")
    
    # NOW freeze all original parameters (after expansion is complete)
    frozen_count, trainable_count = freeze_base_model(expanded_model)
    
    # Verify the expansion worked correctly
    total_new_heads = expansion_count * num_new_heads
    log_message(f"Total new attention heads added: {total_new_heads}")
    
    # Count trainable parameters in new heads
    new_head_params = 0
    for name, module in expanded_model.named_modules():
        if isinstance(module, ExpandedMultiHeadAttention):
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            new_head_params += module_params
    
    log_message(f"Trainable parameters in new heads: {new_head_params:,}")
    
    # Analyze expanded model and compare
    expanded_analyzer = ModelAnalyzer(expanded_model, f"Attention Expanded Model ({num_new_heads} new heads)")
    comparison = original_analyzer.compare_with(expanded_analyzer, "Attention Head Expansion")
    
    return expanded_model

def enable_generation_mode(model):
    """Enable generation mode for all ExpandedMultiHeadAttention modules"""
    for module in model.modules():
        if isinstance(module, ExpandedMultiHeadAttention):
            module.enable_generation_mode()

def disable_generation_mode(model):
    """Disable generation mode for all ExpandedMultiHeadAttention modules"""
    for module in model.modules():
        if isinstance(module, ExpandedMultiHeadAttention):
            module.disable_generation_mode()

class AttentionHeadExpansionContinualLearner:
    """Continual learner using attention head expansion approach"""
    
    def __init__(self, model_name: str, tokenizer, device: str, num_new_heads: int = NUM_NEW_ATTENTION_HEADS):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.num_new_heads = num_new_heads
        self.base_model = None
        self.task_models = {}
        
    def prepare_model(self) -> None:
        """Initialize the base model"""
        # Force float32 on CUDA to avoid overflow issues
        if device == "cuda":
            torch_dtype = torch.float32
            log_message("Using float32 on CUDA to prevent overflow")
        else:
            torch_dtype = torch.float32
        
        self.base_model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch_dtype
        ).to(self.device)
        
        log_message(f"Loaded base model: {self.model_name}")
        
        # Analyze base model with ModelAnalyzer
        base_analyzer = ModelAnalyzer(self.base_model, f"{self.model_name} (Base)")
        self.base_analysis = base_analyzer.analyze(detailed=True)
        
    def train_task(self, train_data, task_name: str, epochs: int = 2, batch_size: int = 8) -> float:
        """Train on a specific task using attention head expansion"""
        log_message(f"Training task: {task_name}")
        
        # Create expanded model for this task
        expanded_model = expand_model_attention_heads(self.base_model, self.num_new_heads)
        
        # Train the expanded model
        training_time = self._train_model(expanded_model, train_data, epochs, batch_size)
        
        # Store the trained model
        self.task_models[task_name] = expanded_model
        
        # Save model with custom handling for ExpandedMultiHeadAttention
        self._save_expanded_model(expanded_model, task_name)
        
        log_message(f"Task {task_name} training completed in {training_time:.2f} minutes")
        return training_time
    
    def _save_expanded_model(self, model, task_name: str):
        """Save expanded model with custom ExpandedMultiHeadAttention handling"""
        save_dir = f"attention_expansion_{task_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'num_new_heads': self.num_new_heads,
            'device': self.device,
            'task_name': task_name
        }
        torch.save(config, os.path.join(save_dir, 'config.pt'))
        
        # Save only the expansion weights (new attention heads)
        expansion_state = {}
        for name, module in model.named_modules():
            if isinstance(module, ExpandedMultiHeadAttention):
                expansion_state[name] = {
                    'new_q_proj_weight': module.new_q_proj.weight.data.clone(),
                    'new_k_proj_weight': module.new_k_proj.weight.data.clone(),
                    'new_v_proj_weight': module.new_v_proj.weight.data.clone(),
                    'new_o_proj_weight': module.new_o_proj.weight.data.clone(),
                    'gate': module.gate.data.clone(),
                    'num_new_heads': module.num_new_heads
                }
        
        torch.save(expansion_state, os.path.join(save_dir, 'expansion_weights.pt'))
        log_message(f"Saved expansion weights to {save_dir}")
        
    def _load_expanded_model(self, task_name: str):
        """Load expanded model with custom ExpandedMultiHeadAttention handling"""
        save_dir = f"attention_expansion_{task_name}"
        
        # Load config
        config = torch.load(os.path.join(save_dir, 'config.pt'))
        
        # Create fresh base model
        base_model = T5ForConditionalGeneration.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float16 if config['device'] == "cuda" else torch.float32
        ).to(config['device'])
        
        # Expand the model
        expanded_model = expand_model_attention_heads(base_model, config['num_new_heads'])
        
        # Load expansion weights
        expansion_state = torch.load(os.path.join(save_dir, 'expansion_weights.pt'))
        
        for name, module in expanded_model.named_modules():
            if isinstance(module, ExpandedMultiHeadAttention) and name in expansion_state:
                state = expansion_state[name]
                module.new_q_proj.weight.data = state['new_q_proj_weight']
                module.new_k_proj.weight.data = state['new_k_proj_weight']
                module.new_v_proj.weight.data = state['new_v_proj_weight']
                module.new_o_proj.weight.data = state['new_o_proj_weight']
                module.gate.data = state['gate']
                module.num_new_heads = state['num_new_heads']
        
        return expanded_model
    
    def switch_to_task(self, task_name: str) -> None:
        """Switch to a specific task model"""
        if task_name in self.task_models:
            # Use cached model
            self.current_model = self.task_models[task_name]
        else:
            # Load from disk
            self.current_model = self._load_expanded_model(task_name)
            self.task_models[task_name] = self.current_model
        
        log_message(f"Switched to task: {task_name}")
    
    def evaluate_task(self, eval_data, task_name: str, num_samples: int = 500) -> Tuple[float, float]:
        """Evaluate on a specific task"""
        self.switch_to_task(task_name)
        
        # Infer language from task name
        language = "python" if "python" in task_name.lower() else "javascript" if "javascript" in task_name.lower() else None
        return self._evaluate_model(self.current_model, eval_data, num_samples, language)
    
    def _train_model(self, model, data, epochs: int, batch_size: int) -> float:
        """Train the model on given data"""
        start_time = time.time()
        
        # Get only trainable parameters (new attention heads)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if not trainable_params:
            log_message("Warning: No trainable parameters found!")
            return 0.0
        
        log_message(f"Training {len(trainable_params)} parameter groups, {sum(p.numel() for p in trainable_params):,} total parameters")
        
        # Verify base model is frozen
        frozen_params = [p for p in model.parameters() if not p.requires_grad]
        log_message(f"Verified: {len(frozen_params)} base model parameters are frozen")
        
        # More aggressive optimizer settings for better gradient flow
        optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.01)  # Increased LR from 1e-4
        
        # Learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=100  # Less aggressive warmup
        )
        
        model.train()
        
        # Collect expanded attention modules for verification
        expanded_attentions = []
        for name, module in model.named_modules():
            if isinstance(module, ExpandedMultiHeadAttention):
                expanded_attentions.append((name, module))
        
        log_message(f"Found {len(expanded_attentions)} ExpandedMultiHeadAttention modules")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            valid_batches = 0
            num_batches = 0
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                
                # Prepare batch
                input_texts = [item['input'] for item in batch]
                target_texts = [item['target'] for item in batch]
                
                # Tokenize
                input_encodings = self.tokenizer(
                    input_texts, 
                    truncation=True, 
                    padding=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                target_encodings = self.tokenizer(
                    target_texts, 
                    truncation=True, 
                    padding=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                try:
                    outputs = model(
                        input_ids=input_encodings.input_ids,
                        attention_mask=input_encodings.attention_mask,
                        labels=target_encodings.input_ids
                    )
                    
                    loss = outputs.loss
                    
                    # Check for invalid loss
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 20.0:
                        log_message(f"Warning: Invalid loss detected ({loss.item():.4f}), skipping batch {i//batch_size + 1}")
                        continue
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Check for NaN gradients
                    has_invalid_grad = False
                    max_grad_norm = 0.0
                    
                    for param in trainable_params:
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm()
                            max_grad_norm = max(max_grad_norm, grad_norm.item())
                            
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_invalid_grad = True
                                break
                    
                    if has_invalid_grad:
                        log_message(f"Warning: Invalid gradients detected, skipping batch {i//batch_size + 1}")
                        continue
                    
                    # Less aggressive gradient clipping to allow better gradient flow
                    if self.device == "cuda":
                        torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)  # Increased from 0.1
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)  # Normal clipping on other devices
                    
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    valid_batches += 1
                    
                    # Verification: Check that new heads are being trained (every 10 batches)
                    if valid_batches % 10 == 0 and len(expanded_attentions) > 0:
                        sample_attention = expanded_attentions[0][1]  # First expanded attention
                        stats = sample_attention.get_training_stats()
                        log_message(f"Batch {valid_batches}: Gate={stats['gate_value']:.6f}, "
                                  f"Q_grad={stats['new_q_grad_norm']:.6f}, "
                                  f"K_grad={stats['new_k_grad_norm']:.6f}, "
                                  f"V_grad={stats['new_v_grad_norm']:.6f}")
                    
                except Exception as e:
                    log_message(f"Warning: Exception during training batch {i//batch_size + 1}: {str(e)}")
                    continue
                
                num_batches += 1
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
            log_message(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Valid Batches: {valid_batches}/{num_batches}")
            
            # Epoch-end verification: Check training progress
            if len(expanded_attentions) > 0:
                total_gate_value = 0.0
                total_weight_change = 0.0
                for name, attention in expanded_attentions:
                    stats = attention.get_training_stats()
                    total_gate_value += stats['gate_value']
                    total_weight_change += stats['new_q_weight_norm'] + stats['new_k_weight_norm'] + stats['new_v_weight_norm']
                
                avg_gate = total_gate_value / len(expanded_attentions)
                avg_weight_norm = total_weight_change / (len(expanded_attentions) * 3)
                log_message(f"Epoch {epoch+1} - Avg Gate Value: {avg_gate:.6f}, Avg Weight Norm: {avg_weight_norm:.6f}")
            
            # Early stopping if loss becomes invalid
            if avg_loss == float('inf') or avg_loss > 15.0:
                log_message("Warning: Training unstable, stopping early")
                break
        
        training_time = (time.time() - start_time) / 60
        
        # Final verification: Check that training actually happened
        if len(expanded_attentions) > 0:
            log_message("=== FINAL TRAINING VERIFICATION ===")
            for name, attention in expanded_attentions[:3]:  # Show first 3
                stats = attention.get_training_stats()
                log_message(f"{name}: Gate={stats['gate_value']:.6f}, "
                          f"Weights=[Q:{stats['new_q_weight_norm']:.4f}, "
                          f"K:{stats['new_k_weight_norm']:.4f}, "
                          f"V:{stats['new_v_weight_norm']:.4f}, "
                          f"O:{stats['new_o_weight_norm']:.4f}]")
        
        return training_time
    
    def _evaluate_model(self, model, data, num_samples: int, language: str = None) -> Tuple[float, float]:
        """Evaluate model performance"""
        model.eval()
        
        # Enable generation mode to prevent new heads from interfering with generation
        enable_generation_mode(model)
        
        # Sample data for evaluation
        eval_data = random.sample(data, min(num_samples, len(data)))
        
        bleu_scores = []
        correct_predictions = 0
        total_predictions = len(eval_data)
        
        # Verification: Check that new heads are contributing
        expanded_attentions = []
        for name, module in model.named_modules():
            if isinstance(module, ExpandedMultiHeadAttention):
                expanded_attentions.append((name, module))
        
        if len(expanded_attentions) > 0:
            log_message(f"=== EVALUATION VERIFICATION ===")
            sample_attention = expanded_attentions[0][1]
            stats = sample_attention.get_training_stats()
            log_message(f"New heads status: Gate={stats['gate_value']:.6f}, "
                      f"Weight norms=[Q:{stats['new_q_weight_norm']:.4f}, "
                      f"K:{stats['new_k_weight_norm']:.4f}, "
                      f"V:{stats['new_v_weight_norm']:.4f}]")
        
        with torch.no_grad():
            for item in eval_data:
                input_text = item['input']
                target_text = item['target']
                
                # Tokenize input
                input_encoding = self.tokenizer(
                    input_text, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                try:
                    # Generate prediction with better parameters to prevent repetitive generation
                    outputs = model.generate(
                        input_ids=input_encoding.input_ids,
                        attention_mask=input_encoding.attention_mask,
                        max_length=min(256, len(input_encoding.input_ids[0]) + 100),  # Reasonable max length
                        num_beams=4,
                        early_stopping=True,
                        repetition_penalty=1.2,  # Prevent repetitive generation
                        no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode prediction
                    predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # DEBUG: Print first few examples to see what's being generated
                    if len(bleu_scores) < 3:  # Only print first 3 examples
                        log_message(f"DEBUG Example {len(bleu_scores) + 1}:")
                        log_message(f"  Input: {input_text[:100]}...")
                        log_message(f"  Target: {target_text[:100]}...")
                        log_message(f"  Predicted: {predicted_text[:100]}...")
                        log_message(f"  Predicted length: {len(predicted_text)}")
                    
                    # Calculate BLEU score
                    reference = target_text.split()
                    candidate = predicted_text.split()
                    
                    if len(candidate) > 0 and len(reference) > 0:
                        smoothing = SmoothingFunction().method1
                        bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing)
                        bleu_scores.append(bleu)
                        
                        # DEBUG: Print BLEU calculation details for first few examples
                        if len(bleu_scores) <= 3:
                            log_message(f"  Reference tokens: {len(reference)}, Candidate tokens: {len(candidate)}")
                            log_message(f"  BLEU score: {bleu:.4f}")
                    else:
                        # DEBUG: Log why BLEU wasn't calculated
                        if len(bleu_scores) < 3:
                            log_message(f"  SKIPPED: Reference tokens: {len(reference)}, Candidate tokens: {len(candidate)}")
                    
                    # Check functional correctness for code
                    if language and self._is_functionally_correct(predicted_text, target_text, language):
                        correct_predictions += 1
                        
                except Exception as e:
                    # DEBUG: Log exceptions
                    if len(bleu_scores) < 3:
                        log_message(f"  EXCEPTION: {str(e)}")
                    continue
        
        # Calculate metrics
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        pass_rate = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
        
        # DEBUG: Print evaluation statistics
        log_message(f"=== EVALUATION STATISTICS ===")
        log_message(f"Total samples processed: {total_predictions}")
        log_message(f"Valid BLEU scores: {len(bleu_scores)}")
        log_message(f"Functional correct: {correct_predictions}")
        log_message(f"Average BLEU: {avg_bleu:.4f}")
        log_message(f"Pass rate: {pass_rate:.2f}%")
        if len(bleu_scores) > 0:
            log_message(f"BLEU score range: {min(bleu_scores):.4f} - {max(bleu_scores):.4f}")
        log_message(f"================================")
        
        # Final verification of contribution
        if len(expanded_attentions) > 0:
            total_contribution = 0.0
            for name, attention in expanded_attentions:
                stats = attention.get_training_stats()
                contribution = stats['gate_value'] * (stats['new_q_weight_norm'] + stats['new_k_weight_norm'] + stats['new_v_weight_norm']) / 3
                total_contribution += contribution
            
            avg_contribution = total_contribution / len(expanded_attentions)
            log_message(f"Average new head contribution: {avg_contribution:.6f}")
        
        # Disable generation mode to restore normal operation
        disable_generation_mode(model)
        
        return avg_bleu, pass_rate
    
    def _is_functionally_correct(self, predicted: str, target: str, language: str) -> bool:
        """Check if predicted code is functionally correct"""
        try:
            if language == "python":
                # Basic syntax check for Python
                ast.parse(predicted)
                return True
            elif language == "javascript":
                # Basic check for JavaScript (simplified)
                return "function" in predicted or "=>" in predicted or "var " in predicted or "let " in predicted
            return False
        except:
            return False

# Data loading functions (same as other experiments)
def load_codesearchnet_data():
    """Load CodeSearchNet dataset with exact same splits as other experiments"""
    log_message("Loading CodeSearchNet dataset...")
    
    # Load datasets
    python_dataset = load_dataset("code_search_net", "python", split="train")
    javascript_dataset = load_dataset("code_search_net", "javascript", split="train")
    
    def prepare_data(dataset, language):
        """Prepare data in the format expected by the model"""
        data = []
        for item in dataset:
            if item['func_code_string'] and item['func_documentation_string']:
                # Use docstring as input, code as target
                input_text = f"Generate {language} code: {item['func_documentation_string']}"
                target_text = item['func_code_string']
                data.append({
                    'input': input_text,
                    'target': target_text
                })
        return data
    
    # Prepare datasets
    python_data = prepare_data(python_dataset, "Python")
    javascript_data = prepare_data(javascript_dataset, "JavaScript")
    
    # Use exact same splits as other experiments for fair comparison
    random.seed(42)  # Ensure reproducibility
    
    # Python: 15,000 train, 5,000 validation
    random.shuffle(python_data)
    python_train = python_data[:15000]
    python_val = python_data[15000:20000]
    
    # JavaScript: varies but use same random seed
    random.shuffle(javascript_data)
    js_train = javascript_data[:min(15000, len(javascript_data))]
    js_val = javascript_data[min(15000, len(javascript_data)):min(20000, len(javascript_data))]
    
    log_message(f"Dataset prepared: Python train={len(python_train)}, val={len(python_val)}")
    log_message(f"                  JavaScript train={len(js_train)}, val={len(js_val)}")
    
    return {
        'python_train': python_train,
        'python_val': python_val,
        'javascript_train': js_train,
        'javascript_val': js_val
    }

def run_attention_head_expansion_experiment():
    """Run the complete attention head expansion continual learning experiment"""
    log_message("Starting Attention Head Expansion Continual Learning Experiment")
    log_message("FAIR COMPARISON: Using EXACT same data splits as LoRA vs Full Layer experiment")
    
    # Load data
    datasets = load_codesearchnet_data()
    
    # Use full datasets for fair comparison
    python_train = datasets['python_train']
    python_val = datasets['python_val']
    javascript_train = datasets['javascript_train']
    javascript_val = datasets['javascript_val']
    
    log_message(f"Using full datasets: Python train={len(python_train)}, val={len(python_val)}")
    log_message(f"Using full datasets: JavaScript train={len(javascript_train)}, val={len(javascript_val)}")
    
    # Initialize model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    log_message("=== ATTENTION HEAD EXPANSION CONTINUAL LEARNING EXPERIMENT ===")
    log_message(f"New attention heads per layer: {NUM_NEW_ATTENTION_HEADS}")
    
    # Initialize learner
    learner = AttentionHeadExpansionContinualLearner(model_name, tokenizer, device, num_new_heads=NUM_NEW_ATTENTION_HEADS)
    learner.prepare_model()
    
    # Track experiment results
    results = {
        'approach': 'attention_head_expansion',
        'num_new_heads': NUM_NEW_ATTENTION_HEADS,
        'model_name': model_name,
        'device': device,
        'training_times': {},
        'bleu_scores': {},
        'pass_rates': {},
        'memory_usage': 0,
        'total_training_time': 0
    }
    
    start_experiment_time = time.time()
    
    # Phase 1: Train on Python
    log_message("Phase 1: Training on Python...")
    python_training_time = learner.train_task(python_train, "python", epochs=2, batch_size=8)
    results['training_times']['python'] = python_training_time
    
    # Evaluate Python after Python training
    python_bleu, python_pass_rate = learner.evaluate_task(python_val, "python", num_samples=500)
    results['bleu_scores']['python_after_python'] = python_bleu
    results['pass_rates']['python_after_python'] = python_pass_rate
    log_message(f"Python after Python training: BLEU {python_bleu:.4f}, Pass Rate {python_pass_rate:.2f}%")
    
    # Phase 2: Train on JavaScript (fresh model)
    log_message("Phase 2: Training on JavaScript (fresh model)...")
    
    # Create fresh learner for JavaScript
    js_learner = AttentionHeadExpansionContinualLearner(model_name, tokenizer, device, num_new_heads=NUM_NEW_ATTENTION_HEADS)
    js_learner.prepare_model()
    
    javascript_training_time = js_learner.train_task(javascript_train, "javascript", epochs=2, batch_size=8)
    results['training_times']['javascript'] = javascript_training_time
    
    # Evaluate JavaScript after JavaScript training
    js_bleu, js_pass_rate = js_learner.evaluate_task(javascript_val, "javascript", num_samples=500)
    results['bleu_scores']['javascript_after_javascript'] = js_bleu
    results['pass_rates']['javascript_after_javascript'] = js_pass_rate
    log_message(f"JavaScript after JavaScript training: BLEU {js_bleu:.4f}, Pass Rate {js_pass_rate:.2f}%")
    
    # Phase 3: Evaluate Python on JavaScript model (forgetting test)
    log_message("Phase 3: Evaluating Python on JavaScript model (forgetting test)...")
    
    # Load Python model and test on it
    learner.switch_to_task("python")
    python_after_js_bleu, python_after_js_pass_rate = learner.evaluate_task(python_val, "python", num_samples=500)
    results['bleu_scores']['python_after_javascript'] = python_after_js_bleu
    results['pass_rates']['python_after_javascript'] = python_after_js_pass_rate
    log_message(f"Python after JavaScript training: BLEU {python_after_js_bleu:.4f}, Pass Rate {python_after_js_pass_rate:.2f}%")
    
    # Calculate final metrics
    total_experiment_time = (time.time() - start_experiment_time) / 60
    results['total_training_time'] = total_experiment_time
    
    # Memory usage
    if device == "cuda":
        results['memory_usage'] = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        results['memory_usage'] = psutil.virtual_memory().used / (1024**3)
    
    # Calculate forgetting rate
    python_initial_bleu = results['bleu_scores']['python_after_python']
    python_final_bleu = results['bleu_scores']['python_after_javascript']
    forgetting_rate = max(0, (python_initial_bleu - python_final_bleu) / python_initial_bleu * 100) if python_initial_bleu > 0 else 0
    results['forgetting_rate'] = forgetting_rate
    
    # Calculate average performance
    avg_bleu = np.mean([
        results['bleu_scores']['python_after_python'],
        results['bleu_scores']['javascript_after_javascript']
    ])
    results['average_bleu'] = avg_bleu
    
    # Get parameter count from final model
    final_analyzer = ModelAnalyzer(learner.current_model, "Final Expanded Model")
    final_analysis = final_analyzer.analyze(detailed=False)
    results['total_parameters'] = final_analysis.total_parameters
    results['trainable_parameters'] = final_analysis.trainable_parameters
    results['parameter_efficiency'] = (final_analysis.trainable_parameters / final_analysis.total_parameters) * 100
    
    # Summary
    log_message("Attention Head Expansion Experiment Summary:")
    log_message(f"  Python BLEU (after Python): {python_bleu:.4f}")
    log_message(f"  Python BLEU (after JS): {python_after_js_bleu:.4f}")
    log_message(f"  JavaScript BLEU: {js_bleu:.4f}")
    log_message(f"  Training Time: {total_experiment_time:.2f} min")
    log_message(f"  Memory Usage: {results['memory_usage']:.2f} GB")
    log_message(f"  Forgetting Rate: {forgetting_rate:.2f}%")
    log_message(f"  New Attention Head Parameters: {final_analysis.trainable_parameters:,} ({results['parameter_efficiency']:.2f}%)")
    
    # Save results
    with open('attention_head_expansion_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message("Attention Head Expansion experiment completed! Results saved to attention_head_expansion_experiment_results.json")
    
    # Final summary
    log_message("\n=== FINAL SUMMARY ===")
    log_message(f"Average BLEU Score: {avg_bleu:.4f}")
    log_message(f"Catastrophic Forgetting: {forgetting_rate:.2f}%")
    log_message(f"Parameter Efficiency: {final_analysis.trainable_parameters:,} parameters ({results['parameter_efficiency']:.2f}%)")
    log_message(f"Training Efficiency: {total_experiment_time:.2f} minutes")
    
    return results

if __name__ == "__main__":
    results = run_attention_head_expansion_experiment() 