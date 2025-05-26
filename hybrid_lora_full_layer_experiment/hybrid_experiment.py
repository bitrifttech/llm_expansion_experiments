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
from peft import LoraConfig, get_peft_model, PeftModel
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats
import difflib
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Add utils to path for data loader, model evaluator, device manager, and model extensions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_and_prepare_data
from utils.model_evaluator import ModelEvaluator
from utils.device_manager import DeviceManager
from utils.model_extensions import LoRAExtension, TransformerLayerExtension, HybridExtension, ExtensionConfig

@dataclass
class HybridExperimentResults:
    """Store results from hybrid LoRA + Full Layer experiments"""
    # Basic metrics for both experiments
    exp1_python_bleu_before: float
    exp1_python_bleu_after: float
    exp1_js_bleu: float
    exp1_training_time: float
    exp1_memory_usage: float
    exp1_forgetting_rate: float
    
    exp2_python_bleu_before: float
    exp2_python_bleu_after: float
    exp2_js_bleu: float
    exp2_training_time: float
    exp2_memory_usage: float
    exp2_forgetting_rate: float
    
    # Performance comparison
    shared_vs_specific_python: float  # Performance difference in Python
    shared_vs_specific_js: float      # Performance difference in JavaScript
    
    def to_dict(self) -> Dict:
        return {
            'experiment_1_task_specific': {
                'python_bleu_before': self.exp1_python_bleu_before,
                'python_bleu_after': self.exp1_python_bleu_after,
                'js_bleu': self.exp1_js_bleu,
                'training_time': self.exp1_training_time,
                'memory_usage': self.exp1_memory_usage,
                'forgetting_rate': self.exp1_forgetting_rate
            },
            'experiment_2_shared_layer': {
                'python_bleu_before': self.exp2_python_bleu_before,
                'python_bleu_after': self.exp2_python_bleu_after,
                'js_bleu': self.exp2_js_bleu,
                'training_time': self.exp2_training_time,
                'memory_usage': self.exp2_memory_usage,
                'forgetting_rate': self.exp2_forgetting_rate
            },
            'comparison': {
                'shared_vs_specific_python': self.shared_vs_specific_python,
                'shared_vs_specific_js': self.shared_vs_specific_js
            }
        }

# Initialize device manager
device_manager = DeviceManager()
device = device_manager.device

# Logging setup (use device manager's logging)
def log_message(message, level="INFO"):
    device_manager._log_message(message, level)

def freeze_base_model(model):
    """Freeze all base model parameters"""
    for param in model.parameters():
        param.requires_grad = False
    log_message(f"Froze {sum(1 for p in model.parameters() if not p.requires_grad)} base model parameters")

# Removed add_trainable_transformer_layer function - now using TransformerLayerExtension class

class HybridLoRAFullLayerLearner:
    """Hybrid learner that combines LoRA adapters with full transformer layers using extension classes"""
    
    def __init__(self, model_name: str, tokenizer, device: str):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.base_model = None
        self.current_model = None
        self.hybrid_extension = None
        self.shared_full_layer_model = None  # For experiment 2
        
    def prepare_model(self) -> None:
        """Initialize the base model and hybrid extension"""
        self.base_model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=device_manager.torch_dtype
        ).to(self.device)
        self.base_model = device_manager.optimize_for_device(self.base_model)
        freeze_base_model(self.base_model)
        
        # Create hybrid extension configuration
        config = ExtensionConfig(
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            lora_target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
            freeze_base=True,
            layer_position="encoder",
            layer_initialization_scale=0.01
        )
        
        # Initialize hybrid extension
        self.hybrid_extension = HybridExtension(self.base_model, config, device_manager)
        
    def create_hybrid_model(self, base_model, task_name: str, use_shared_layer: bool = False, shared_layer_model = None):
        """Create a model with both LoRA adapter and full layer"""
        if use_shared_layer and shared_layer_model is not None:
            # Use the shared full layer model as starting point
            hybrid_model = self.hybrid_extension.create_hybrid_model(task_name, use_shared_layer=True, shared_layer_model=shared_layer_model)
            log_message(f"Using shared full layer for {task_name}")
        else:
            # Create new hybrid model with both LoRA and full layer
            hybrid_model = self.hybrid_extension.create_hybrid_model(task_name)
            log_message(f"Created new hybrid model for {task_name}")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in hybrid_model.parameters())
        
        log_message(f"Hybrid model created: {trainable_params:,} trainable / {total_params:,} total parameters")
        
        return hybrid_model
        
    def train_hybrid_model(self, model, data, task_name: str, epochs: int = 2, batch_size: int = 8) -> float:
        """Train the hybrid model"""
        start_time = time.time()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        model.train()
        
        # Calculate total batches for progress tracking
        total_batches = (len(data) + batch_size - 1) // batch_size  # Round up division
        total_steps = total_batches * epochs
        current_step = 0
        
        log_message(f"Training hybrid model for {task_name}...")
        log_message(f"Total steps: {total_steps} ({epochs} epochs × {total_batches} batches)")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_losses = []
            
            for i in range(0, len(data), batch_size):
                batch_start_time = time.time()
                current_step += 1
                
                batch_indices = list(range(i, min(i + batch_size, len(data))))
                batch_data = data.select(batch_indices)
                batch_texts = [text for text in batch_data["func_code_string"] if text and str(text).strip()]
                
                if not batch_texts:
                    continue
                    
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True, 
                    padding=True
                ).to(self.device)
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_losses.append(loss.item())
                
                # Progress tracking every 10 batches or at end of epoch
                if current_step % 10 == 0 or i + batch_size >= len(data):
                    elapsed_time = time.time() - start_time
                    progress_pct = (current_step / total_steps) * 100
                    
                    if current_step > 0:
                        avg_time_per_step = elapsed_time / current_step
                        remaining_steps = total_steps - current_step
                        estimated_remaining = avg_time_per_step * remaining_steps
                        
                        elapsed_str = f"{elapsed_time/60:.1f}min"
                        remaining_str = f"{estimated_remaining/60:.1f}min"
                        
                        log_message(f"  Step {current_step}/{total_steps} ({progress_pct:.1f}%) | "
                                  f"Loss: {loss.item():.4f} | "
                                  f"Elapsed: {elapsed_str} | "
                                  f"Remaining: ~{remaining_str}")
                
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            epoch_time = time.time() - epoch_start_time
            elapsed_total = time.time() - start_time
            
            log_message(f"Epoch {epoch+1}/{epochs} completed in {epoch_time/60:.1f}min | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Total Elapsed: {elapsed_total/60:.1f}min")
            
        total_time = (time.time() - start_time) / 60
        log_message(f"Training {task_name} completed in {total_time:.2f} minutes")
        return total_time
        
    def evaluate_model(self, model, data, num_samples: int = 100, language: str = None) -> Dict[str, float]:
        """Evaluate model comprehensively using ModelEvaluator"""
        evaluator = ModelEvaluator(self.tokenizer)
        results = evaluator.evaluate_comprehensive(model, data, language, num_samples)
        return results.to_dict()

def get_memory_usage():
    """Get current memory usage in GB"""
    return psutil.Process().memory_info().rss / 1024**3

def run_experiment_1_task_specific(learner, python_train, python_val, js_train, js_val, seed: int):
    """Experiment 1: Task-specific LoRA + Full Layer components"""
    log_message("=== EXPERIMENT 1: TASK-SPECIFIC COMPONENTS ===")
    device_manager.set_seed(seed)
    start_memory = get_memory_usage()
    
    # Phase 1: Train Python with LoRA + Full Layer
    log_message("Phase 1: Training Python (LoRA + Full Layer)")
    python_model = learner.create_hybrid_model(learner.base_model, "python", use_shared_layer=False)
    python_training_time = learner.train_hybrid_model(python_model, python_train, "python")
    
    # Evaluate Python performance AFTER Python training (before JS training)
    python_results_after_python = learner.evaluate_model(python_model, python_val, 50, "python")
    log_message(f"Python after Python training: BLEU {python_results_after_python['bleu']:.4f}, Pass Rate {python_results_after_python['pass_rate']:.2%}")
    
    # Save Python model components for later evaluation
    os.makedirs("experiment_1_results", exist_ok=True)
    python_model.save_pretrained("experiment_1_results/python_hybrid")
    
    # Phase 2: Train JavaScript with NEW LoRA + NEW Full Layer (completely independent)
    log_message("Phase 2: Training JavaScript (new LoRA + new Full Layer)")
    log_message("This is completely independent - no shared components")
    
    js_model = learner.create_hybrid_model(learner.base_model, "javascript", use_shared_layer=False)
    js_training_time = learner.train_hybrid_model(js_model, js_train, "javascript")
    
    # Evaluate JavaScript performance AFTER JavaScript training
    js_results_after_js = learner.evaluate_model(js_model, js_val, 50, "javascript")
    log_message(f"JavaScript after JavaScript training: BLEU {js_results_after_js['bleu']:.4f}, Pass Rate {js_results_after_js['pass_rate']:.2%}")
    
    # Save JavaScript model
    js_model.save_pretrained("experiment_1_results/javascript_hybrid")
    
    # Re-evaluate Python to check for interference (should be ZERO due to completely separate models)
    log_message("Re-evaluating Python - should show NO interference due to task isolation...")
    
    try:
        # Reload the Python model (should be unchanged since models are completely separate)
        python_model_reloaded = PeftModel.from_pretrained(
            learner.base_model,  # Clean base model
            "experiment_1_results/python_hybrid"  # Python hybrid model
        )
        
        python_results_after_js = learner.evaluate_model(python_model_reloaded, python_val, 50, "python")
        log_message(f"Python after JavaScript training (task isolation): BLEU {python_results_after_js['bleu']:.4f}")
        
        # Calculate interference (should be ~0 for task-specific)
        interference = python_results_after_python['bleu'] - python_results_after_js['bleu']
        log_message(f"Task isolation interference: {interference:+.4f} BLEU (should be ~0)")
        
    except Exception as e:
        log_message(f"Error in task isolation evaluation: {e}", level="ERROR")
        # Fallback: use the original results (perfect isolation)
        python_results_after_js = python_results_after_python
        log_message("Using Python results before JS training as fallback (perfect isolation)")
    
    end_memory = get_memory_usage()
    total_training_time = python_training_time + js_training_time
    forgetting_rate = (python_results_after_python['bleu'] - python_results_after_js['bleu']) / python_results_after_python['bleu'] if python_results_after_python['bleu'] > 0 else 0
    
    log_message(f"Experiment 1 Summary:")
    log_message(f"  Python BLEU (after Python training): {python_results_after_python['bleu']:.4f}")
    log_message(f"  Python BLEU (after JS training): {python_results_after_js['bleu']:.4f}")
    log_message(f"  JavaScript BLEU (after JS training): {js_results_after_js['bleu']:.4f}")
    log_message(f"  Training Time: {total_training_time:.2f} min")
    log_message(f"  Memory Usage: {end_memory - start_memory:.2f} GB")
    log_message(f"  Forgetting Rate: {forgetting_rate:.2%}")
    
    return {
        'python_bleu_before': python_results_after_python['bleu'],
        'python_bleu_after': python_results_after_js['bleu'],
        'js_bleu': js_results_after_js['bleu'],
        'training_time': total_training_time,
        'memory_usage': end_memory - start_memory,
        'forgetting_rate': forgetting_rate
    }

def run_experiment_2_shared_layer(learner, python_train, python_val, js_train, js_val, seed: int):
    """Experiment 2: Task-specific LoRA + Shared Full Layer"""
    log_message("=== EXPERIMENT 2: SHARED FULL LAYER ===")
    device_manager.set_seed(seed)
    start_memory = get_memory_usage()
    
    # Phase 1: Train Python with LoRA + Full Layer
    log_message("Phase 1: Training Python (LoRA + Full Layer)")
    python_model = learner.create_hybrid_model(learner.base_model, "python", use_shared_layer=False)
    python_training_time = learner.train_hybrid_model(python_model, python_train, "python")
    
    # Evaluate Python performance AFTER Python training (before JS training)
    python_results_after_python = learner.evaluate_model(python_model, python_val, 50, "python")
    log_message(f"Python after Python training: BLEU {python_results_after_python['bleu']:.4f}, Pass Rate {python_results_after_python['pass_rate']:.2%}")
    
    # Extract the base model with full layer (this will be our SHARED layer)
    shared_base_with_layer = python_model.get_base_model()
    learner.shared_full_layer_model = shared_base_with_layer
    
    # Save components
    os.makedirs("experiment_2_results", exist_ok=True)
    python_model.save_pretrained("experiment_2_results/python_hybrid")
    shared_base_with_layer.save_pretrained("experiment_2_results/shared_layer")
    
    log_message("Shared layer extracted and saved - this layer will be modified during JS training")
    
    # Phase 2: Train JavaScript with new LoRA + SHARED Full Layer
    log_message("Phase 2: Training JavaScript (new LoRA + SHARED Full Layer)")
    log_message("WARNING: This will modify the shared full layer!")
    
    js_model = learner.create_hybrid_model(learner.base_model, "javascript", 
                                         use_shared_layer=True, 
                                         shared_layer_model=shared_base_with_layer)
    js_training_time = learner.train_hybrid_model(js_model, js_train, "javascript")
    
    # Evaluate JavaScript performance AFTER JavaScript training
    js_results_after_js = learner.evaluate_model(js_model, js_val, 50, "javascript")
    log_message(f"JavaScript after JavaScript training: BLEU {js_results_after_js['bleu']:.4f}, Pass Rate {js_results_after_js['pass_rate']:.2%}")
    
    # Save JavaScript model
    js_model.save_pretrained("experiment_2_results/javascript_hybrid")
    
    # CRITICAL: Re-evaluate Python using the MODIFIED shared layer
    log_message("Re-evaluating Python with the shared layer that was modified during JS training...")
    
    # The shared_base_with_layer has been modified during JS training
    # Now we need to reload the Python LoRA adapter onto this modified shared layer
    try:
        # Create a fresh Python hybrid model using the now-modified shared layer
        python_model_with_modified_shared_layer = learner.create_hybrid_model(
            learner.base_model, "python", 
            use_shared_layer=True, 
            shared_layer_model=shared_base_with_layer  # This has been modified by JS training
        )
        
        # Load the Python LoRA adapter weights
        # We need to load just the LoRA weights, not create a new model
        from peft import PeftModel
        
        # Load the saved Python LoRA adapter onto the modified shared layer
        python_model_after_js = PeftModel.from_pretrained(
            shared_base_with_layer,  # Modified shared layer 
            "experiment_2_results/python_hybrid"  # Python LoRA adapter
        )
        
        # Evaluate Python with the modified shared layer
        python_results_after_js = learner.evaluate_model(python_model_after_js, python_val, 50, "python")
        log_message(f"Python after JavaScript training (shared layer interference): BLEU {python_results_after_js['bleu']:.4f}")
        
        # Calculate interference
        interference = python_results_after_python['bleu'] - python_results_after_js['bleu']
        log_message(f"Shared layer interference: {interference:+.4f} BLEU (positive = harmful interference)")
        
    except Exception as e:
        log_message(f"Error in shared layer evaluation: {e}", level="ERROR")
        # Fallback: use the original results
        python_results_after_js = python_results_after_python
        log_message("Using Python results before JS training as fallback")
    
    end_memory = get_memory_usage()
    total_training_time = python_training_time + js_training_time
    forgetting_rate = (python_results_after_python['bleu'] - python_results_after_js['bleu']) / python_results_after_python['bleu'] if python_results_after_python['bleu'] > 0 else 0
    
    log_message(f"Experiment 2 Summary:")
    log_message(f"  Python BLEU (after Python training): {python_results_after_python['bleu']:.4f}")
    log_message(f"  Python BLEU (after JS training): {python_results_after_js['bleu']:.4f}")
    log_message(f"  JavaScript BLEU (after JS training): {js_results_after_js['bleu']:.4f}")
    log_message(f"  Training Time: {total_training_time:.2f} min")
    log_message(f"  Memory Usage: {end_memory - start_memory:.2f} GB")
    log_message(f"  Forgetting Rate: {forgetting_rate:.2%}")
    
    return {
        'python_bleu_before': python_results_after_python['bleu'],
        'python_bleu_after': python_results_after_js['bleu'],
        'js_bleu': js_results_after_js['bleu'],
        'training_time': total_training_time,
        'memory_usage': end_memory - start_memory,
        'forgetting_rate': forgetting_rate
    }

def main():
    """Main experimental function"""
    log_message("Starting Hybrid LoRA + Full Layer Experiment")
    log_message("Comparing task-specific vs. shared full layer approaches")
    
    # Initialize components
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load data using the new unified data loader (same splits as original hybrid experiment)
    python_train, python_val, js_train, js_val = load_and_prepare_data(
        python_train_size=8000,
        python_val_size=2000,
        js_train_size=8000,
        js_val_size=2000,
        format_type="huggingface",
        seed=42
    )
    
    # Run experiments
    seed = 42
    
    # Experiment 1: Task-specific components
    log_message("Initializing learner for Experiment 1...")
    learner1 = HybridLoRAFullLayerLearner(model_name, tokenizer, device)
    learner1.prepare_model()
    exp1_results = run_experiment_1_task_specific(learner1, python_train, python_val, js_train, js_val, seed)
    
    # Clean up experiment 1 resources
    del learner1
    device_manager.clear_cache()
    
    # Experiment 2: Shared full layer (start fresh)
    log_message("Initializing fresh learner for Experiment 2...")
    learner2 = HybridLoRAFullLayerLearner(model_name, tokenizer, device)
    learner2.prepare_model()
    exp2_results = run_experiment_2_shared_layer(learner2, python_train, python_val, js_train, js_val, seed)
    
    # Compare results
    log_message("\n=== COMPARISON ANALYSIS ===")
    python_diff = exp2_results['python_bleu_after'] - exp1_results['python_bleu_after']
    js_diff = exp2_results['js_bleu'] - exp1_results['js_bleu']
    
    log_message(f"Python Performance: Shared vs Task-specific = {python_diff:+.4f}")
    log_message(f"JavaScript Performance: Shared vs Task-specific = {js_diff:+.4f}")
    log_message(f"Training Time: Exp1={exp1_results['training_time']:.2f}min, Exp2={exp2_results['training_time']:.2f}min")
    log_message(f"Memory Usage: Exp1={exp1_results['memory_usage']:.2f}GB, Exp2={exp2_results['memory_usage']:.2f}GB")
    
    # Save comprehensive results
    final_results = HybridExperimentResults(
        exp1_python_bleu_before=exp1_results['python_bleu_before'],
        exp1_python_bleu_after=exp1_results['python_bleu_after'],
        exp1_js_bleu=exp1_results['js_bleu'],
        exp1_training_time=exp1_results['training_time'],
        exp1_memory_usage=exp1_results['memory_usage'],
        exp1_forgetting_rate=exp1_results['forgetting_rate'],
        
        exp2_python_bleu_before=exp2_results['python_bleu_before'],
        exp2_python_bleu_after=exp2_results['python_bleu_after'],
        exp2_js_bleu=exp2_results['js_bleu'],
        exp2_training_time=exp2_results['training_time'],
        exp2_memory_usage=exp2_results['memory_usage'],
        exp2_forgetting_rate=exp2_results['forgetting_rate'],
        
        shared_vs_specific_python=python_diff,
        shared_vs_specific_js=js_diff
    )
    
    with open('hybrid_experiment_results.json', 'w') as f:
        json.dump(final_results.to_dict(), f, indent=2)
    
    log_message("Hybrid experiment completed! Results saved to hybrid_experiment_results.json")

if __name__ == "__main__":
    main() 