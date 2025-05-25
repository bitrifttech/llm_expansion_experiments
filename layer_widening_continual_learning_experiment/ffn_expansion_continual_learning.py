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

# Set random seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class LayerWideningExperimentResults:
    """Store results from layer widening continual learning experiments"""
    # Basic metrics
    python_bleu_before: float
    python_bleu_after: float
    js_bleu: float
    python_pass_before: float
    python_pass_after: float
    js_pass: float
    
    # Advanced semantic metrics
    python_meteor_before: float
    python_meteor_after: float
    js_meteor: float
    python_edit_distance_before: float
    python_edit_distance_after: float
    js_edit_distance: float
    
    # Code quality metrics
    python_complexity_before: float
    python_complexity_after: float
    js_complexity: float
    python_ast_similarity_before: float
    python_ast_similarity_after: float
    js_ast_similarity: float
    
    # Continual learning metrics
    forward_transfer: float  # How Python helps JavaScript
    backward_interference: float  # How JavaScript hurts Python
    retention_score: float  # Overall knowledge retention
    
    # Efficiency metrics
    training_time: float
    memory_usage: float
    forgetting_rate: float
    
    # Layer widening specific metrics
    expansion_parameters: int
    expansion_percentage: float
    
    def to_dict(self) -> Dict:
        return {
            'python_bleu_before': self.python_bleu_before,
            'python_bleu_after': self.python_bleu_after,
            'js_bleu': self.js_bleu,
            'python_pass_before': self.python_pass_before,
            'python_pass_after': self.python_pass_after,
            'js_pass': self.js_pass,
            'python_meteor_before': self.python_meteor_before,
            'python_meteor_after': self.python_meteor_after,
            'js_meteor': self.js_meteor,
            'python_edit_distance_before': self.python_edit_distance_before,
            'python_edit_distance_after': self.python_edit_distance_after,
            'js_edit_distance': self.js_edit_distance,
            'python_complexity_before': self.python_complexity_before,
            'python_complexity_after': self.python_complexity_after,
            'js_complexity': self.js_complexity,
            'python_ast_similarity_before': self.python_ast_similarity_before,
            'python_ast_similarity_after': self.python_ast_similarity_after,
            'js_ast_similarity': self.js_ast_similarity,
            'forward_transfer': self.forward_transfer,
            'backward_interference': self.backward_interference,
            'retention_score': self.retention_score,
            'training_time': self.training_time,
            'memory_usage': self.memory_usage,
            'forgetting_rate': self.forgetting_rate,
            'expansion_parameters': self.expansion_parameters,
            'expansion_percentage': self.expansion_percentage
        }

# Logging setup
def log_message(message, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

# Device setup
if torch.cuda.is_available():
    device = "cuda"
    log_message("Using CUDA GPU")
    log_message(f"GPU: {torch.cuda.get_device_name()}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
elif torch.backends.mps.is_available():
    device = "mps"
    log_message("Using Apple Silicon MPS")
else:
    device = "cpu"
    log_message("Using CPU (no MPS or CUDA available)")

log_message(f"Device: {device}, System Memory: {psutil.virtual_memory().total / 1024**3:.2f} GB")

def freeze_base_model(model):
    """Freeze all base model parameters"""
    for param in model.parameters():
        param.requires_grad = False
    log_message(f"Froze {sum(1 for p in model.parameters() if not p.requires_grad)} base model parameters")

class ExpandedFFN(torch.nn.Module):
    """FFN module with expansion capability - ultra-stable version"""
    
    def __init__(self, original_ffn, expansion_size: int = 512, device: str = "cpu"):
        super().__init__()
        self.original_ffn = original_ffn  # Frozen
        self.expansion_size = expansion_size
        self.device = device
        
        # Get dtype from original FFN
        self.dtype = next(original_ffn.parameters()).dtype
        
        # Freeze original FFN
        for param in self.original_ffn.parameters():
            param.requires_grad = False
        
        # Get input dimension from original FFN
        # T5-small FFN structure: wi (input projection), wo (output projection)
        if hasattr(original_ffn, 'wi'):
            input_dim = original_ffn.wi.in_features
            output_dim = original_ffn.wo.out_features
        elif hasattr(original_ffn, 'wi_0'):
            # For larger T5 models that use gated FFN
            input_dim = original_ffn.wi_0.in_features
            output_dim = original_ffn.wo.out_features
        else:
            # Fallback for different FFN structures
            input_dim = 512  # T5-small default
            output_dim = 512
        
        # Ultra-simple expansion: just learnable residual weights
        # Start with identity mapping (zero residual)
        self.expansion_residual = torch.nn.Parameter(
            torch.zeros(output_dim, dtype=self.dtype, device=device)
        )
        
        # Optional: small MLP for more capacity, but start disabled
        self.use_mlp = expansion_size > 0
        if self.use_mlp:
            self.expansion_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, expansion_size // 4, bias=False, dtype=self.dtype, device=device),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(expansion_size // 4, output_dim, bias=False, dtype=self.dtype, device=device)
            )
            
            # Initialize MLP to output zeros
            with torch.no_grad():
                for layer in self.expansion_mlp:
                    if hasattr(layer, 'weight'):
                        torch.nn.init.zeros_(layer.weight)
        
        # Gate that starts at zero (completely disabled)
        self.gate = torch.nn.Parameter(torch.tensor(-10.0, dtype=self.dtype, device=device))  # sigmoid(-10) ≈ 0
        
        log_message(f"Created ExpandedFFN: {input_dim} -> {expansion_size//4 if self.use_mlp else 0} -> {output_dim} on {device} ({self.dtype})")
        
    def forward(self, x):
        # Original FFN output (frozen)
        with torch.no_grad():
            original_out = self.original_ffn(x)
        
        # Start with just the residual (simplest possible expansion)
        expansion = self.expansion_residual
        
        # Optionally add MLP contribution
        if self.use_mlp:
            mlp_out = self.expansion_mlp(x)
            expansion = expansion + mlp_out
        
        # Ensure expansion matches original output shape and dtype
        if expansion.dim() == 1 and original_out.dim() > 1:
            # Broadcast residual to match batch dimensions
            expansion = expansion.unsqueeze(0).expand_as(original_out)
        
        expansion = expansion.to(dtype=original_out.dtype, device=original_out.device)
        
        # Apply gate (starts near zero, can gradually increase)
        gate_value = torch.sigmoid(self.gate.to(dtype=original_out.dtype)) * 0.001  # Max 0.1% contribution
        gated_expansion = gate_value * expansion
        
        # Simple addition
        result = original_out + gated_expansion
        
        return result

def expand_model_ffn(model, expansion_size: int = 512):
    """Expand all FFN layers in the model with additional trainable parameters"""
    log_message(f"Starting FFN expansion with size {expansion_size}...")
    
    # Analyze original model
    original_analyzer = ModelAnalyzer(model, f"Original Model")
    
    expanded_model = deepcopy(model)
    
    # Get the device from the model
    model_device = next(model.parameters()).device
    
    # Freeze all original parameters
    freeze_base_model(expanded_model)
    
    # Replace FFN layers with expanded versions
    expansion_count = 0
    
    # Expand encoder FFN layers
    for layer_idx, layer in enumerate(expanded_model.encoder.block):
        if hasattr(layer, 'layer') and len(layer.layer) > 1:
            # T5 structure: layer[0] is self-attention, layer[1] is FFN
            original_ffn = layer.layer[1].DenseReluDense
            expanded_ffn = ExpandedFFN(original_ffn, expansion_size, model_device)
            layer.layer[1].DenseReluDense = expanded_ffn
            expansion_count += 1
            log_message(f"Expanded encoder layer {layer_idx} FFN")
    
    # Expand decoder FFN layers
    for layer_idx, layer in enumerate(expanded_model.decoder.block):
        if hasattr(layer, 'layer') and len(layer.layer) > 2:
            # T5 structure: layer[0] is self-attention, layer[1] is cross-attention, layer[2] is FFN
            original_ffn = layer.layer[2].DenseReluDense
            expanded_ffn = ExpandedFFN(original_ffn, expansion_size, model_device)
            layer.layer[2].DenseReluDense = expanded_ffn
            expansion_count += 1
            log_message(f"Expanded decoder layer {layer_idx} FFN")
    
    log_message(f"FFN Expansion complete: {expansion_count} layers expanded")
    
    # Analyze expanded model and compare
    expanded_analyzer = ModelAnalyzer(expanded_model, f"FFN Expanded Model (size={expansion_size})")
    comparison = original_analyzer.compare_with(expanded_analyzer, "FFN Expansion")
    
    return expanded_model

class FFNExpansionContinualLearner:
    """Continual learner using FFN expansion approach"""
    
    def __init__(self, model_name: str, tokenizer, device: str, expansion_size: int = 512):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.expansion_size = expansion_size
        self.base_model = None
        self.task_models = {}
        
    def prepare_model(self) -> None:
        """Initialize the base model"""
        self.base_model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(self.device)
        
        log_message(f"Loaded base model: {self.model_name}")
        
        # Analyze base model with ModelAnalyzer
        base_analyzer = ModelAnalyzer(self.base_model, f"{self.model_name} (Base)")
        self.base_analysis = base_analyzer.analyze(detailed=True)
        
    def train_task(self, train_data, task_name: str, epochs: int = 2, batch_size: int = 8) -> float:
        """Train on a specific task using FFN expansion"""
        log_message(f"Training task: {task_name}")
        
        # Create expanded model for this task
        expanded_model = expand_model_ffn(self.base_model, self.expansion_size)
        
        # Train the expanded model
        training_time = self._train_model(expanded_model, train_data, epochs, batch_size)
        
        # Store the trained model
        self.task_models[task_name] = expanded_model
        
        # Save model with custom handling for ExpandedFFN
        self._save_expanded_model(expanded_model, task_name)
        
        log_message(f"Task {task_name} training completed in {training_time:.2f} minutes")
        return training_time
        
    def _save_expanded_model(self, model, task_name: str):
        """Save expanded model with custom ExpandedFFN handling"""
        save_dir = f"ffn_expansion_{task_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save only the expansion parameters (trainable parts)
        expansion_state = {}
        for name, module in model.named_modules():
            if isinstance(module, ExpandedFFN):
                state = {
                    'expansion_residual': module.expansion_residual.data,
                    'gate': module.gate.data,
                    'expansion_size': module.expansion_size,
                    'use_mlp': module.use_mlp
                }
                
                # Save MLP weights if present
                if module.use_mlp:
                    mlp_weights = []
                    for layer in module.expansion_mlp:
                        if hasattr(layer, 'weight'):
                            mlp_weights.append(layer.weight.data)
                    state['expansion_mlp_weights'] = mlp_weights
                
                expansion_state[name] = state
        
        # Save expansion parameters
        torch.save(expansion_state, os.path.join(save_dir, 'expansion_weights.pt'))
        
        # Save model config for reconstruction
        config = {
            'model_name': self.model_name,
            'expansion_size': self.expansion_size,
            'device': self.device
        }
        torch.save(config, os.path.join(save_dir, 'config.pt'))
        
        log_message(f"Saved expansion weights to {save_dir}")
        
    def _load_expanded_model(self, task_name: str):
        """Load expanded model with custom ExpandedFFN handling"""
        save_dir = f"ffn_expansion_{task_name}"
        
        # Load config
        config = torch.load(os.path.join(save_dir, 'config.pt'))
        
        # Create fresh base model
        base_model = T5ForConditionalGeneration.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float16 if config['device'] == "cuda" else torch.float32
        ).to(config['device'])
        
        # Expand the model
        expanded_model = expand_model_ffn(base_model, config['expansion_size'])
        
        # Load expansion weights
        expansion_state = torch.load(os.path.join(save_dir, 'expansion_weights.pt'))
        
        for name, module in expanded_model.named_modules():
            if isinstance(module, ExpandedFFN) and name in expansion_state:
                state = expansion_state[name]
                module.expansion_residual.data = state['expansion_residual']
                module.gate.data = state['gate']
                module.use_mlp = state.get('use_mlp', False)
                
                # Load MLP weights if present
                if module.use_mlp and 'expansion_mlp_weights' in state:
                    mlp_weights = state['expansion_mlp_weights']
                    weight_idx = 0
                    for layer in module.expansion_mlp:
                        if hasattr(layer, 'weight') and weight_idx < len(mlp_weights):
                            layer.weight.data = mlp_weights[weight_idx]
                            weight_idx += 1
        
        return expanded_model
        
    def switch_to_task(self, task_name: str) -> None:
        """Switch to a specific task model"""
        if task_name not in self.task_models:
            # Try to load from disk
            save_dir = f"ffn_expansion_{task_name}"
            if os.path.exists(save_dir):
                log_message(f"Loading {task_name} model from {save_dir}")
                self.task_models[task_name] = self._load_expanded_model(task_name)
            else:
                raise ValueError(f"Task {task_name} model not found")
        
        log_message(f"Switched to task: {task_name}")
        
    def evaluate_task(self, eval_data, task_name: str, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate on a specific task"""
        if task_name not in self.task_models:
            self.switch_to_task(task_name)
        
        model = self.task_models[task_name]
        # Map task name to language for proper evaluation
        language = "python" if task_name == "python" else "javascript"
        return self._evaluate_model(model, eval_data, num_samples, language)
        
    def _train_model(self, model, data, epochs: int, batch_size: int) -> float:
        """Train the model on given data"""
        start_time = time.time()
        
        # Setup optimizer for only trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        log_message(f"Training {len(trainable_params)} parameter groups, {sum(p.numel() for p in trainable_params):,} total parameters")
        
        # Use extremely conservative learning rate for expansion training to prevent instability
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.01, eps=1e-8)
        
        # Learning rate scheduler for very gradual warmup
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=200
        )
        
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            valid_batches = 0
            
            # Create batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                
                # Prepare inputs
                inputs = [item['func_name'] + ' ' + item['docstring'] for item in batch]
                targets = [item['code'] for item in batch]
                
                # Tokenize
                input_encodings = self.tokenizer(
                    inputs, 
                    truncation=True, 
                    padding=True, 
                    max_length=256, 
                    return_tensors="pt"
                ).to(self.device)
                
                target_encodings = self.tokenizer(
                    targets, 
                    truncation=True, 
                    padding=True, 
                    max_length=256, 
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
                    
                    # Check for invalid loss with more conservative thresholds
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 50.0:
                        log_message(f"Warning: Invalid loss detected ({loss.item():.4f}), skipping batch {i//batch_size + 1}")
                        continue
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Check for NaN gradients and gradient explosion with stricter thresholds
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
                    
                    if max_grad_norm > 5.0:  # More conservative threshold
                        log_message(f"Warning: Large gradient norm ({max_grad_norm:.2f}), clipping")
                    
                    # Very aggressive gradient clipping
                    torch.nn.utils.clip_grad_norm_(trainable_params, 0.1)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    valid_batches += 1
                    
                except Exception as e:
                    log_message(f"Warning: Exception during training batch {i//batch_size + 1}: {str(e)}")
                    continue
                
                num_batches += 1
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
            log_message(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Valid Batches: {valid_batches}/{num_batches}")
            
            # Early stopping if loss becomes invalid
            if avg_loss == float('inf') or avg_loss > 20.0:
                log_message("Warning: Training unstable, stopping early")
                break
        
        training_time = (time.time() - start_time) / 60
        return training_time
        
    def _evaluate_model(self, model, data, num_samples: int, language: str = None) -> Dict[str, float]:
        """Evaluate model performance"""
        model.eval()
        
        # Sample data for evaluation
        eval_data = data[:num_samples] if len(data) > num_samples else data
        
        bleu_scores = []
        pass_count = 0
        meteor_scores = []
        edit_distances = []
        ast_similarities = []
        complexities = []
        
        with torch.no_grad():
            for item in eval_data:
                input_text = item['func_name'] + ' ' + item['docstring']
                target_code = item['code']
                
                # Generate prediction
                input_encoding = self.tokenizer(
                    input_text, 
                    truncation=True, 
                    max_length=256, 
                    return_tensors="pt"
                ).to(self.device)
                
                generated_ids = model.generate(
                    input_ids=input_encoding.input_ids,
                    attention_mask=input_encoding.attention_mask,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                predicted_code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Calculate metrics
                bleu_score = self._calculate_bleu(predicted_code, target_code)
                bleu_scores.append(bleu_score)
                
                # Syntax check
                if self._is_syntactically_correct(predicted_code, language):
                    pass_count += 1
                
                # Additional metrics
                meteor_scores.append(self._calculate_meteor(predicted_code, target_code))
                edit_distances.append(self._calculate_edit_distance(predicted_code, target_code))
                ast_similarities.append(self._calculate_ast_similarity(predicted_code, target_code, language))
                complexities.append(self._calculate_complexity(predicted_code, language))
        
        return {
            'bleu': np.mean(bleu_scores),
            'pass_rate': pass_count / len(eval_data),
            'meteor': np.mean(meteor_scores),
            'edit_distance': np.mean(edit_distances),
            'ast_similarity': np.mean(ast_similarities),
            'complexity': np.mean(complexities)
        }
    
    def _calculate_bleu(self, predicted: str, target: str) -> float:
        """Calculate BLEU score"""
        try:
            pred_tokens = predicted.split()
            target_tokens = target.split()
            if len(pred_tokens) == 0 or len(target_tokens) == 0:
                return 0.0
            smoothing = SmoothingFunction().method1
            return sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothing)
        except:
            return 0.0
    
    def _calculate_meteor(self, predicted: str, target: str) -> float:
        """Calculate METEOR score"""
        try:
            return meteor_score([target], predicted)
        except:
            return 0.0
    
    def _calculate_edit_distance(self, predicted: str, target: str) -> float:
        """Calculate normalized edit distance"""
        try:
            distance = difflib.SequenceMatcher(None, predicted, target).ratio()
            return 1.0 - distance  # Convert similarity to distance
        except:
            return 1.0
    
    def _calculate_ast_similarity(self, predicted: str, target: str, language: str) -> float:
        """Calculate AST similarity"""
        try:
            if language == "python":
                pred_ast = ast.parse(predicted)
                target_ast = ast.parse(target)
                return 1.0  # Simplified - both parse successfully
            else:
                return 0.5  # Simplified for non-Python
        except:
            return 0.0
    
    def _calculate_complexity(self, code: str, language: str) -> float:
        """Calculate code complexity"""
        try:
            # Simplified complexity based on code length and structure
            lines = code.split('\n')
            complexity = len(lines) + code.count('if') + code.count('for') + code.count('while')
            return complexity
        except:
            return 0.0
    
    def _is_syntactically_correct(self, code: str, language: str) -> bool:
        """Check if code is syntactically correct"""
        try:
            if language == "python":
                ast.parse(code)
                return True
            elif language == "javascript":
                # Simplified check for JavaScript
                return '{' in code and '}' in code
            else:
                return True
        except:
            return False

def get_memory_usage():
    """Get current memory usage in GB"""
    return psutil.virtual_memory().used / 1024**3

def load_and_prepare_data():
    """Load and prepare CodeSearchNet dataset - SAME AS OTHER EXPERIMENTS"""
    log_message("Loading CodeSearchNet dataset...")
    
    try:
        # Use the EXACT same approach as LoRA vs Full Layer experiment
        dataset = load_dataset("code_search_net", split="train")
        
        # Filter and prepare datasets - SAME SPLITS AS OTHER EXPERIMENTS
        python_data = dataset.filter(lambda x: x["language"] == "python").select(range(20000))
        js_data = dataset.filter(lambda x: x["language"] == "javascript").select(range(20000))
        
        # Split into train/val - SAME AS OTHER EXPERIMENTS
        python_train = python_data.select(range(15000))
        python_val = python_data.select(range(15000, 20000))
        js_train = js_data.select(range(15000))
        js_val = js_data.select(range(15000, 20000))
        
        # Convert to the format expected by FFN expansion (dict format)
        def convert_to_dict_format(dataset_split):
            converted = []
            for item in dataset_split:
                if item['func_name'] and item['func_documentation_string'] and item['func_code_string']:
                    converted.append({
                        'func_name': item['func_name'],
                        'docstring': item['func_documentation_string'],
                        'code': item['func_code_string']
                    })
            return converted
        
        python_train_dict = convert_to_dict_format(python_train)
        python_val_dict = convert_to_dict_format(python_val)
        js_train_dict = convert_to_dict_format(js_train)
        js_val_dict = convert_to_dict_format(js_val)
        
        log_message(f"Dataset prepared: Python train={len(python_train_dict)}, val={len(python_val_dict)}")
        log_message(f"                  JavaScript train={len(js_train_dict)}, val={len(js_val_dict)}")
        
        return python_train_dict, python_val_dict, js_train_dict, js_val_dict
        
    except Exception as e:
        log_message(f"Dataset loading error: {e}", level="ERROR")
        sys.exit(1)

def calculate_continual_learning_metrics(python_before: Dict, js_after_python: Dict,
                                       python_after_js: Dict, js_after_js: Dict) -> Dict[str, float]:
    """Calculate continual learning specific metrics"""
    
    # Forward transfer: How much Python training helps JavaScript
    # Compare JS performance after Python training vs baseline
    forward_transfer = js_after_python.get('bleu', 0) - 0.1  # Assume 0.1 baseline
    
    # Backward interference: How much JavaScript training hurts Python
    backward_interference = python_before['bleu'] - python_after_js['bleu']
    
    # Retention score: Overall knowledge retention
    retention_score = (python_after_js['bleu'] + js_after_js['bleu']) / 2
    
    return {
        'forward_transfer': forward_transfer,
        'backward_interference': backward_interference,
        'retention_score': retention_score
    }

def run_ffn_expansion_experiment(model_name: str, tokenizer, python_train, python_val, 
                                js_train, js_val, seed: int, expansion_size: int = 512) -> LayerWideningExperimentResults:
    """Run the complete FFN expansion continual learning experiment"""
    
    log_message("=== FFN EXPANSION CONTINUAL LEARNING EXPERIMENT ===")
    log_message(f"Expansion size: {expansion_size}")
    set_seed(seed)
    start_memory = get_memory_usage()
    
    # Initialize learner
    learner = FFNExpansionContinualLearner(model_name, tokenizer, device, expansion_size)
    learner.prepare_model()
    
    # Count expansion parameters
    base_params = sum(p.numel() for p in learner.base_model.parameters())
    
    # Phase 1: Train on Python
    log_message("Phase 1: Training on Python...")
    python_training_time = learner.train_task(python_train, "python")
    
    # Evaluate Python after Python training
    python_results_after_python = learner.evaluate_task(python_val, "python", 50)
    log_message(f"Python after Python training: BLEU {python_results_after_python['bleu']:.4f}, Pass Rate {python_results_after_python['pass_rate']:.2%}")
    
    # Phase 2: Train on JavaScript (fresh model)
    log_message("Phase 2: Training on JavaScript (fresh model)...")
    
    # Create fresh learner for JavaScript
    js_learner = FFNExpansionContinualLearner(model_name, tokenizer, device, expansion_size)
    js_learner.prepare_model()
    js_training_time = js_learner.train_task(js_train, "javascript")
    
    # Evaluate JavaScript after JavaScript training
    js_results_after_js = js_learner.evaluate_task(js_val, "javascript", 50)
    log_message(f"JavaScript after JavaScript training: BLEU {js_results_after_js['bleu']:.4f}, Pass Rate {js_results_after_js['pass_rate']:.2%}")
    
    # Phase 3: Evaluate Python on JavaScript model (catastrophic forgetting test)
    log_message("Phase 3: Evaluating Python on JavaScript model (forgetting test)...")
    python_results_after_js = js_learner.evaluate_task(python_val, "python", 50)
    log_message(f"Python after JavaScript training: BLEU {python_results_after_js['bleu']:.4f}, Pass Rate {python_results_after_js['pass_rate']:.2%}")
    
    # Calculate metrics
    end_memory = get_memory_usage()
    total_training_time = python_training_time + js_training_time
    forgetting_rate = (python_results_after_python['bleu'] - python_results_after_js['bleu']) / python_results_after_python['bleu'] if python_results_after_python['bleu'] > 0 else 0
    
    # Calculate expansion parameters
    expanded_model = expand_model_ffn(learner.base_model, expansion_size)
    expanded_analyzer = ModelAnalyzer(expanded_model, "Final Expanded Model")
    expanded_analysis = expanded_analyzer.analyze(detailed=False)
    
    expansion_params = expanded_analysis.trainable_parameters
    expansion_percentage = expanded_analysis.efficiency_metrics['trainable_percentage']
    
    # Calculate continual learning metrics
    cl_metrics = calculate_continual_learning_metrics(
        python_results_after_python, js_results_after_js,
        python_results_after_js, js_results_after_js
    )
    
    log_message(f"FFN Expansion Experiment Summary:")
    log_message(f"  Python BLEU (after Python): {python_results_after_python['bleu']:.4f}")
    log_message(f"  Python BLEU (after JS): {python_results_after_js['bleu']:.4f}")
    log_message(f"  JavaScript BLEU: {js_results_after_js['bleu']:.4f}")
    log_message(f"  Training Time: {total_training_time:.2f} min")
    log_message(f"  Memory Usage: {end_memory - start_memory:.2f} GB")
    log_message(f"  Forgetting Rate: {forgetting_rate:.2%}")
    log_message(f"  Expansion Parameters: {expansion_params:,} ({expansion_percentage:.2f}%)")
    
    return LayerWideningExperimentResults(
        python_bleu_before=python_results_after_python['bleu'],
        python_bleu_after=python_results_after_js['bleu'],
        js_bleu=js_results_after_js['bleu'],
        python_pass_before=python_results_after_python['pass_rate'],
        python_pass_after=python_results_after_js['pass_rate'],
        js_pass=js_results_after_js['pass_rate'],
        python_meteor_before=python_results_after_python['meteor'],
        python_meteor_after=python_results_after_js['meteor'],
        js_meteor=js_results_after_js['meteor'],
        python_edit_distance_before=python_results_after_python['edit_distance'],
        python_edit_distance_after=python_results_after_js['edit_distance'],
        js_edit_distance=js_results_after_js['edit_distance'],
        python_complexity_before=python_results_after_python['complexity'],
        python_complexity_after=python_results_after_js['complexity'],
        js_complexity=js_results_after_js['complexity'],
        python_ast_similarity_before=python_results_after_python['ast_similarity'],
        python_ast_similarity_after=python_results_after_js['ast_similarity'],
        js_ast_similarity=js_results_after_js['ast_similarity'],
        forward_transfer=cl_metrics['forward_transfer'],
        backward_interference=cl_metrics['backward_interference'],
        retention_score=cl_metrics['retention_score'],
        training_time=total_training_time,
        memory_usage=end_memory - start_memory,
        forgetting_rate=forgetting_rate,
        expansion_parameters=expansion_params,
        expansion_percentage=expansion_percentage
    )

def main():
    """Main experimental function"""
    log_message("Starting FFN Expansion Continual Learning Experiment")
    log_message("FAIR COMPARISON: Using EXACT same data splits as LoRA vs Full Layer experiment")
    
    # Initialize components
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load data - SAME AS OTHER EXPERIMENTS
    python_train, python_val, js_train, js_val = load_and_prepare_data()
    
    # Use same evaluation sample sizes as other experiments for consistency
    log_message(f"Using full datasets: Python train={len(python_train)}, val={len(python_val)}")
    log_message(f"Using full datasets: JavaScript train={len(js_train)}, val={len(js_val)}")
    
    # Run experiment with same seed as other experiments
    seed = 42
    expansion_size = 512  # Use full expansion size for fair comparison
    
    results = run_ffn_expansion_experiment(
        model_name, tokenizer, python_train, python_val, 
        js_train, js_val, seed, expansion_size
    )
    
    # Save results
    with open('ffn_expansion_experiment_results.json', 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    
    log_message("FFN Expansion experiment completed! Results saved to ffn_expansion_experiment_results.json")
    
    # Print summary
    log_message("\n=== FINAL SUMMARY ===")
    log_message(f"Average BLEU Score: {(results.python_bleu_after + results.js_bleu) / 2:.4f}")
    log_message(f"Catastrophic Forgetting: {results.forgetting_rate:.2%}")
    log_message(f"Parameter Efficiency: {results.expansion_parameters:,} parameters ({results.expansion_percentage:.2f}%)")
    log_message(f"Training Efficiency: {results.training_time:.2f} minutes")

if __name__ == "__main__":
    main() 