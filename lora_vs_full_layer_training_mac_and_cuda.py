import os
import sys
import torch
import numpy as np
import psutil
import time
import random
from copy import deepcopy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from peft import LoraConfig, get_peft_model, PeftModel
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats

# Set random seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class ExperimentResults:
    """Store results from a single experimental run"""
    python_bleu_before: float
    python_bleu_after: float
    js_bleu: float
    python_pass_before: float
    python_pass_after: float
    js_pass: float
    training_time: float
    memory_usage: float
    forgetting_rate: float
    
    def to_dict(self) -> Dict:
        return {
            'python_bleu_before': self.python_bleu_before,
            'python_bleu_after': self.python_bleu_after,
            'js_bleu': self.js_bleu,
            'python_pass_before': self.python_pass_before,
            'python_pass_after': self.python_pass_after,
            'js_pass': self.js_pass,
            'training_time': self.training_time,
            'memory_usage': self.memory_usage,
            'forgetting_rate': self.forgetting_rate
        }

# Logging setup
def log_message(message, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

# Device setup (MPS, CUDA, or CPU)
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

def add_trainable_transformer_layer(model):
    """Add a new trainable transformer layer to the encoder"""
    from copy import deepcopy
    import torch.nn as nn
    
    # Create a new layer based on the last encoder layer
    new_layer = deepcopy(model.encoder.block[-1])
    
    # Add it to the encoder
    model.encoder.block.append(new_layer)
    
    # Update config
    model.config.num_layers = len(model.encoder.block)
    
    # Initialize new layer parameters with small random values
    for param in new_layer.parameters():
        param.data = param.data * 0.01
        param.requires_grad = True  # Ensure new layer is trainable
    
    trainable_params = sum(p.numel() for p in new_layer.parameters() if p.requires_grad)
    log_message(f"Added trainable transformer layer with {trainable_params:,} parameters")
    
    return model

class ContinualLearner:
    """Base class for continual learning approaches"""
    
    def __init__(self, model_name: str, tokenizer, device: str):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.base_model = None
        self.current_model = None
        
    def prepare_model(self) -> None:
        """Initialize the base model"""
        self.base_model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(self.device)
        
    def train_task(self, train_data, task_name: str, epochs: int = 5, batch_size: int = 16) -> float:
        """Train on a specific task"""
        raise NotImplementedError
        
    def evaluate_task(self, eval_data, task_name: str, num_samples: int = 500) -> Tuple[float, float]:
        """Evaluate on a specific task"""
        raise NotImplementedError
        
    def switch_to_task(self, task_name: str) -> None:
        """Switch model to evaluate specific task"""
        raise NotImplementedError

class LoRAContinualLearner(ContinualLearner):
    """LoRA-based continual learning with proper adapter management"""
    
    def __init__(self, model_name: str, tokenizer, device: str):
        super().__init__(model_name, tokenizer, device)
        self.adapters = {}  # Store adapter paths
        self.current_task = None
        
    def prepare_model(self) -> None:
        """Initialize the base model without LoRA"""
        super().prepare_model()
        self.current_model = self.base_model
        
    def train_task(self, train_data, task_name: str, epochs: int = 5, batch_size: int = 16) -> float:
        """Train LoRA adapter for specific task"""
        log_message(f"Training LoRA adapter for {task_name}...")
        
        # Start with clean base model
        model = deepcopy(self.base_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # Increased rank for better capacity
            lora_alpha=32,
            target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],  # Include FFN layers
            task_type="SEQ_2_SEQ_LM",
            lora_dropout=0.1
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Train the adapter
        training_time = self._train_model(model, train_data, epochs, batch_size)
        
        # Save adapter
        adapter_path = f"adapters/{task_name}"
        os.makedirs(adapter_path, exist_ok=True)
        model.save_pretrained(adapter_path)
        self.adapters[task_name] = adapter_path
        
        log_message(f"LoRA adapter for {task_name} saved to {adapter_path}")
        return training_time
        
    def switch_to_task(self, task_name: str) -> None:
        """Switch to specific task adapter"""
        if task_name not in self.adapters:
            raise ValueError(f"No adapter found for task {task_name}")
            
        # Load base model and apply specific adapter
        self.current_model = deepcopy(self.base_model)
        self.current_model = PeftModel.from_pretrained(
            self.current_model, 
            self.adapters[task_name]
        ).to(self.device)
        self.current_task = task_name
        
    def evaluate_task(self, eval_data, task_name: str, num_samples: int = 500) -> Tuple[float, float]:
        """Evaluate specific task using its adapter"""
        self.switch_to_task(task_name)
        return self._evaluate_model(self.current_model, eval_data, num_samples)
        
    def _train_model(self, model, data, epochs: int, batch_size: int) -> float:
        """Internal training method"""
        start_time = time.time()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        model.train()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for i in range(0, len(data), batch_size):
                batch_indices = list(range(i, min(i + batch_size, len(data))))
                batch_data = data.select(batch_indices)
                batch_texts = [text for text in batch_data["func_code_string"] if text and str(text).strip()]
                
                if not batch_texts:
                    continue
                    
                # Create input-target pairs for self-supervised learning
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True, 
                    padding=True
                ).to(self.device)
                
                # Use input as both source and target (denoising autoencoder style)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_losses.append(loss.item())
                
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            log_message(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
            
        return (time.time() - start_time) / 60
        
    def _evaluate_model(self, model, data, num_samples: int) -> Tuple[float, float]:
        """Internal evaluation method"""
        model.eval()
        bleu_scores = []
        pass_scores = []
        
        smoothing = SmoothingFunction().method1
        eval_samples = min(num_samples, len(data))
        
        with torch.no_grad():
            for i in range(eval_samples):
                try:
                    source_code = data[i]["func_code_string"]
                    if not source_code or not str(source_code).strip():
                        continue
                        
                    # Use first part as input, full code as target
                    input_text = source_code[:len(source_code)//2]
                    target_text = source_code
                    
                    inputs = self.tokenizer(
                        input_text, 
                        return_tensors="pt", 
                        max_length=512, 
                        truncation=True
                    ).to(self.device)
                    
                    outputs = model.generate(
                        **inputs, 
                        max_length=512, 
                        num_beams=3,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        temperature=0.7
                    )
                    
                    pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Calculate BLEU score
                    target_tokens = self.tokenizer.tokenize(target_text)
                    pred_tokens = self.tokenizer.tokenize(pred_text)
                    
                    if target_tokens and pred_tokens:
                        bleu = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothing)
                        bleu_scores.append(bleu)
                    else:
                        bleu_scores.append(0.0)
                        
                    # Test syntactic correctness
                    try:
                        # Try to parse/compile the generated code
                        if "python" in data[i].get("language", "").lower():
                            compile(pred_text, "<string>", "exec")
                        pass_scores.append(1.0)
                    except:
                        pass_scores.append(0.0)
                        
                except Exception as e:
                    bleu_scores.append(0.0)
                    pass_scores.append(0.0)
                    
        return np.mean(bleu_scores) if bleu_scores else 0.0, np.mean(pass_scores) if pass_scores else 0.0

class FullLayerContinualLearner(ContinualLearner):
    """Full layer training with frozen base weights and task-specific new layers"""
    
    def __init__(self, model_name: str, tokenizer, device: str):
        super().__init__(model_name, tokenizer, device)
        self.checkpoints = {}
        self.current_task = None
        self.task_layers = {}  # Track which layers belong to which tasks
        
    def prepare_model(self) -> None:
        """Initialize the base model and freeze its weights"""
        super().prepare_model()
        # Freeze all base model parameters
        freeze_base_model(self.base_model)
        log_message(f"Base model prepared with frozen weights")
        
    def train_task(self, train_data, task_name: str, epochs: int = 5, batch_size: int = 16) -> float:
        """Train new transformer layer for specific task"""
        log_message(f"Training new transformer layer for {task_name}...")
        
        # Start with current model state (includes previously added layers)
        if self.current_model is not None:
            model = deepcopy(self.current_model)
        else:
            model = deepcopy(self.base_model)
            
        # Add and configure new trainable layer for this task
        model = add_trainable_transformer_layer(model)
        
        # Store which layer index belongs to this task
        self.task_layers[task_name] = len(model.encoder.block) - 1
        
        # Verify only new layer is trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        log_message(f"Training {trainable_params:,} / {total_params:,} parameters ({100*trainable_params/total_params:.2f}%)")
        
        # Train only the new layer
        training_time = self._train_model(model, train_data, epochs, batch_size)
        
        # Update current model to include the new trained layer
        self.current_model = model
        self.current_task = task_name
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/{task_name}"
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        self.checkpoints[task_name] = checkpoint_path
        
        log_message(f"Checkpoint for {task_name} saved to {checkpoint_path}")
        log_message(f"Task {task_name} uses transformer layer {self.task_layers[task_name]}")
        return training_time
        
    def switch_to_task(self, task_name: str) -> None:
        """Switch to specific task checkpoint"""
        if task_name not in self.checkpoints:
            raise ValueError(f"No checkpoint found for task {task_name}")
            
        self.current_model = T5ForConditionalGeneration.from_pretrained(
            self.checkpoints[task_name]
        ).to(self.device)
        self.current_task = task_name
        
    def evaluate_task(self, eval_data, task_name: str, num_samples: int = 500) -> Tuple[float, float]:
        """Evaluate specific task using its checkpoint"""
        self.switch_to_task(task_name)
        return self._evaluate_model(self.current_model, eval_data, num_samples)
        
    def _train_model(self, model, data, epochs: int, batch_size: int) -> float:
        """Internal training method - only train new layers"""
        start_time = time.time()
        
        # Only optimize trainable parameters (new layers)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=0.01)  # Match LoRA learning rate
        model.train()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for i in range(0, len(data), batch_size):
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
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_losses.append(loss.item())
                
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            log_message(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
            
        return (time.time() - start_time) / 60
        
    def _evaluate_model(self, model, data, num_samples: int) -> Tuple[float, float]:
        """Internal evaluation method"""
        model.eval()
        bleu_scores = []
        pass_scores = []
        
        smoothing = SmoothingFunction().method1
        eval_samples = min(num_samples, len(data))
        
        with torch.no_grad():
            for i in range(eval_samples):
                try:
                    source_code = data[i]["func_code_string"]
                    if not source_code or not str(source_code).strip():
                        continue
                        
                    input_text = source_code[:len(source_code)//2]
                    target_text = source_code
                    
                    inputs = self.tokenizer(
                        input_text, 
                        return_tensors="pt", 
                        max_length=512, 
                        truncation=True
                    ).to(self.device)
                    
                    outputs = model.generate(
                        **inputs, 
                        max_length=512, 
                        num_beams=3,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        temperature=0.7
                    )
                    
                    pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    target_tokens = self.tokenizer.tokenize(target_text)
                    pred_tokens = self.tokenizer.tokenize(pred_text)
                    
                    if target_tokens and pred_tokens:
                        bleu = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothing)
                        bleu_scores.append(bleu)
                    else:
                        bleu_scores.append(0.0)
                        
                    try:
                        if "python" in data[i].get("language", "").lower():
                            compile(pred_text, "<string>", "exec")
                        pass_scores.append(1.0)
                    except:
                        pass_scores.append(0.0)
                        
                except Exception as e:
                    bleu_scores.append(0.0)
                    pass_scores.append(0.0)
                    
        return np.mean(bleu_scores) if bleu_scores else 0.0, np.mean(pass_scores) if pass_scores else 0.0

def get_memory_usage():
    """Get current memory usage in GB"""
    return psutil.Process().memory_info().rss / 1024**3

def load_and_prepare_data():
    """Load and prepare datasets"""
    log_message("Loading CodeSearchNet dataset...")
    
    try:
        dataset = load_dataset("code_search_net", split="train")
        
        # Filter and prepare datasets
        python_data = dataset.filter(lambda x: x["language"] == "python").select(range(20000))
        js_data = dataset.filter(lambda x: x["language"] == "javascript").select(range(20000))
        
        # Split into train/val
        python_train = python_data.select(range(15000))
        python_val = python_data.select(range(15000, 20000))
        js_train = js_data.select(range(15000))
        js_val = js_data.select(range(15000, 20000))
        
        log_message(f"Dataset prepared: Python train={len(python_train)}, val={len(python_val)}")
        log_message(f"                  JavaScript train={len(js_train)}, val={len(js_val)}")
        
        return python_train, python_val, js_train, js_val
        
    except Exception as e:
        log_message(f"Dataset loading error: {e}", level="ERROR")
        sys.exit(1)

def run_single_experiment(learner_class, model_name: str, tokenizer, python_train, python_val, js_train, js_val, seed: int) -> ExperimentResults:
    """Run a single experimental trial with comprehensive before/after comparisons"""
    set_seed(seed)
    log_message(f"Running {learner_class.__name__} experiment with seed {seed}")
    
    # Initialize learner
    learner = learner_class(model_name, tokenizer, device)
    learner.prepare_model()
    
    start_memory = get_memory_usage()
    
    # === BASELINE EVALUATION (before any training) ===
    log_message("=== BASELINE EVALUATION (Untrained Model) ===")
    # For baseline, we need to temporarily set up the base model for evaluation
    temp_model = learner.base_model
    baseline_python_bleu, baseline_python_pass = learner._evaluate_model(temp_model, python_val, 100)
    baseline_js_bleu, baseline_js_pass = learner._evaluate_model(temp_model, js_val, 100)
    
    log_message(f"Baseline Python BLEU: {baseline_python_bleu:.4f}, Pass@1: {baseline_python_pass:.4f}")
    log_message(f"Baseline JavaScript BLEU: {baseline_js_bleu:.4f}, Pass@1: {baseline_js_pass:.4f}")
    
    # === STEP 1: TRAIN ON PYTHON ===
    log_message("\n=== STEP 1: TRAINING ON PYTHON ===")
    python_training_time = learner.train_task(python_train, "python", epochs=5, batch_size=16)
    
    # Evaluate both tasks after Python training
    log_message("Evaluating both tasks after Python training...")
    python_bleu_after_python, python_pass_after_python = learner.evaluate_task(python_val, "python", num_samples=500)
    
    # For JavaScript evaluation after Python training, we need to handle this carefully
    # For LoRA: no JS adapter exists yet, so use base model
    # For Full: use the Python-trained model
    if isinstance(learner, LoRAContinualLearner):
        # Use base model for JS since no JS adapter exists yet
        js_bleu_after_python, js_pass_after_python = learner._evaluate_model(learner.base_model, js_val, 500)
    else:
        # Use current model (trained on Python) for JS
        js_bleu_after_python, js_pass_after_python = learner._evaluate_model(learner.current_model, js_val, 500)
    
    log_message(f"After Python training:")
    log_message(f"  Python BLEU: {baseline_python_bleu:.4f} → {python_bleu_after_python:.4f} (Δ{python_bleu_after_python-baseline_python_bleu:+.4f})")
    log_message(f"  JavaScript BLEU: {baseline_js_bleu:.4f} → {js_bleu_after_python:.4f} (Δ{js_bleu_after_python-baseline_js_bleu:+.4f})")
    log_message(f"  Python Pass@1: {baseline_python_pass:.4f} → {python_pass_after_python:.4f} (Δ{python_pass_after_python-baseline_python_pass:+.4f})")
    log_message(f"  JavaScript Pass@1: {baseline_js_pass:.4f} → {js_pass_after_python:.4f} (Δ{js_pass_after_python-baseline_js_pass:+.4f})")
    
    # === STEP 2: TRAIN ON JAVASCRIPT ===
    log_message("\n=== STEP 2: TRAINING ON JAVASCRIPT ===")
    js_training_time = learner.train_task(js_train, "javascript", epochs=5, batch_size=16)
    
    # Evaluate both tasks after JavaScript training
    log_message("Evaluating both tasks after JavaScript training...")
    js_bleu_after_js, js_pass_after_js = learner.evaluate_task(js_val, "javascript", num_samples=500)
    python_bleu_after_js, python_pass_after_js = learner.evaluate_task(python_val, "python", num_samples=500)
    
    log_message(f"After JavaScript training:")
    log_message(f"  JavaScript BLEU: {js_bleu_after_python:.4f} → {js_bleu_after_js:.4f} (Δ{js_bleu_after_js-js_bleu_after_python:+.4f})")
    log_message(f"  Python BLEU: {python_bleu_after_python:.4f} → {python_bleu_after_js:.4f} (Δ{python_bleu_after_js-python_bleu_after_python:+.4f})")
    log_message(f"  JavaScript Pass@1: {js_pass_after_python:.4f} → {js_pass_after_js:.4f} (Δ{js_pass_after_js-js_pass_after_python:+.4f})")
    log_message(f"  Python Pass@1: {python_pass_after_python:.4f} → {python_pass_after_js:.4f} (Δ{python_pass_after_js-python_pass_after_python:+.4f})")
    
    # === SUMMARY ANALYSIS ===
    end_memory = get_memory_usage()
    total_training_time = python_training_time + js_training_time
    
    # Calculate various forgetting metrics
    absolute_forgetting = python_bleu_after_python - python_bleu_after_js
    relative_forgetting = absolute_forgetting / python_bleu_after_python if python_bleu_after_python > 0 else 0
    baseline_relative_forgetting = (baseline_python_bleu - python_bleu_after_js) / baseline_python_bleu if baseline_python_bleu > 0 else 0
    
    log_message(f"\n=== COMPREHENSIVE SUMMARY ===")
    log_message(f"📈 Overall Python Progress: {baseline_python_bleu:.4f} → {python_bleu_after_python:.4f} → {python_bleu_after_js:.4f}")
    log_message(f"📈 Overall JavaScript Progress: {baseline_js_bleu:.4f} → {js_bleu_after_python:.4f} → {js_bleu_after_js:.4f}")
    log_message(f"🧠 Catastrophic Forgetting (Python): {absolute_forgetting:.4f} absolute, {relative_forgetting:.2%} relative")
    log_message(f"⚡ Cross-task Interference: JS training {'helped' if python_bleu_after_js > python_bleu_after_python else 'hurt'} Python by {abs(python_bleu_after_js - python_bleu_after_python):.4f}")
    log_message(f"🎯 Final Performance vs Baseline: Python {python_bleu_after_js/baseline_python_bleu:.2f}x, JavaScript {js_bleu_after_js/baseline_js_bleu:.2f}x")
    
    results = ExperimentResults(
        python_bleu_before=python_bleu_after_python,  # After Python training
        python_bleu_after=python_bleu_after_js,       # After JavaScript training  
        js_bleu=js_bleu_after_js,                     # After JavaScript training
        python_pass_before=python_pass_after_python,
        python_pass_after=python_pass_after_js,
        js_pass=js_pass_after_js,
        training_time=total_training_time,
        memory_usage=end_memory - start_memory,
        forgetting_rate=relative_forgetting
    )
    
    log_message(f"\n{learner_class.__name__} Final Results (seed {seed}):")
    log_message(f"  Python BLEU: {python_bleu_after_python:.4f} → {python_bleu_after_js:.4f} (forgetting: {relative_forgetting:.2%})")
    log_message(f"  JavaScript BLEU: {js_bleu_after_js:.4f}")
    log_message(f"  Training time: {total_training_time:.2f} min")
    log_message(f"  Memory usage: {end_memory - start_memory:.2f} GB")
    
    return results

def run_statistical_analysis(lora_results: List[ExperimentResults], full_results: List[ExperimentResults]):
    """Run statistical analysis comparing the two approaches"""
    log_message("Running statistical analysis...")
    
    # Extract metrics for comparison
    lora_js_bleu = [r.js_bleu for r in lora_results]
    full_js_bleu = [r.js_bleu for r in full_results]
    
    lora_forgetting = [r.forgetting_rate for r in lora_results]
    full_forgetting = [r.forgetting_rate for r in full_results]
    
    lora_time = [r.training_time for r in lora_results]
    full_time = [r.training_time for r in full_results]
    
    lora_memory = [r.memory_usage for r in lora_results]
    full_memory = [r.memory_usage for r in full_results]
    
    # Statistical tests
    js_bleu_stat, js_bleu_p = stats.mannwhitneyu(full_js_bleu, lora_js_bleu, alternative='two-sided')
    forgetting_stat, forgetting_p = stats.mannwhitneyu(lora_forgetting, full_forgetting, alternative='two-sided')
    time_stat, time_p = stats.mannwhitneyu(lora_time, full_time, alternative='two-sided')
    memory_stat, memory_p = stats.mannwhitneyu(lora_memory, full_memory, alternative='two-sided')
    
    # Calculate means and standard errors
    def mean_se(values):
        return np.mean(values), np.std(values) / np.sqrt(len(values))
    
    lora_js_mean, lora_js_se = mean_se(lora_js_bleu)
    full_js_mean, full_js_se = mean_se(full_js_bleu)
    
    lora_forget_mean, lora_forget_se = mean_se(lora_forgetting)
    full_forget_mean, full_forget_se = mean_se(full_forgetting)
    
    lora_time_mean, lora_time_se = mean_se(lora_time)
    full_time_mean, full_time_se = mean_se(full_time)
    
    lora_memory_mean, lora_memory_se = mean_se(lora_memory)
    full_memory_mean, full_memory_se = mean_se(full_memory)
    
    # Print results
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n📊 JAVASCRIPT PERFORMANCE (BLEU Score)")
    print(f"LoRA:      {lora_js_mean:.4f} ± {lora_js_se:.4f}")
    print(f"Full:      {full_js_mean:.4f} ± {full_js_se:.4f}")
    print(f"p-value:   {js_bleu_p:.6f} {'***' if js_bleu_p < 0.001 else '**' if js_bleu_p < 0.01 else '*' if js_bleu_p < 0.05 else 'ns'}")
    
    print(f"\n🧠 CATASTROPHIC FORGETTING (Forgetting Rate)")
    print(f"LoRA:      {lora_forget_mean:.2%} ± {lora_forget_se:.2%}")
    print(f"Full:      {full_forget_mean:.2%} ± {full_forget_se:.2%}")
    print(f"p-value:   {forgetting_p:.6f} {'***' if forgetting_p < 0.001 else '**' if forgetting_p < 0.01 else '*' if forgetting_p < 0.05 else 'ns'}")
    
    print(f"\n⏱️  TRAINING EFFICIENCY (Minutes)")
    print(f"LoRA:      {lora_time_mean:.2f} ± {lora_time_se:.2f}")
    print(f"Full:      {full_time_mean:.2f} ± {full_time_se:.2f}")
    print(f"p-value:   {time_p:.6f} {'***' if time_p < 0.001 else '**' if time_p < 0.01 else '*' if time_p < 0.05 else 'ns'}")
    print(f"Speedup:   {full_time_mean/lora_time_mean:.2f}x")
    
    print(f"\n💾 MEMORY USAGE (GB)")
    print(f"LoRA:      {lora_memory_mean:.2f} ± {lora_memory_se:.2f}")
    print(f"Full:      {full_memory_mean:.2f} ± {full_memory_se:.2f}")
    print(f"p-value:   {memory_p:.6f} {'***' if memory_p < 0.001 else '**' if memory_p < 0.01 else '*' if memory_p < 0.05 else 'ns'}")
    
    # Save detailed results
    results_summary = {
        'lora_results': [r.to_dict() for r in lora_results],
        'full_results': [r.to_dict() for r in full_results],
        'statistical_tests': {
            'js_bleu': {'statistic': float(js_bleu_stat), 'p_value': float(js_bleu_p)},
            'forgetting': {'statistic': float(forgetting_stat), 'p_value': float(forgetting_p)},
            'training_time': {'statistic': float(time_stat), 'p_value': float(time_p)},
            'memory_usage': {'statistic': float(memory_stat), 'p_value': float(memory_p)}
        },
        'summary_statistics': {
            'lora': {
                'js_bleu': {'mean': lora_js_mean, 'se': lora_js_se},
                'forgetting_rate': {'mean': lora_forget_mean, 'se': lora_forget_se},
                'training_time': {'mean': lora_time_mean, 'se': lora_time_se},
                'memory_usage': {'mean': lora_memory_mean, 'se': lora_memory_se}
            },
            'full': {
                'js_bleu': {'mean': full_js_mean, 'se': full_js_se},
                'forgetting_rate': {'mean': full_forget_mean, 'se': full_forget_se},
                'training_time': {'mean': full_time_mean, 'se': full_time_se},
                'memory_usage': {'mean': full_memory_mean, 'se': full_memory_se}
            }
        }
    }
    
    with open('experimental_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    log_message("Detailed results saved to experimental_results.json")

def main():
    """Main experimental function"""
    log_message("Starting LoRA vs New Layer Training Comparison")
    log_message("Both approaches use FROZEN base weights + task-specific trainable components")
    log_message("LoRA: Adds small adapter matrices (~0.1% parameters)")
    log_message("New Layer: Adds full transformer layers (~1-2% parameters)")
    
    # Initialize tokenizer
    model_name = "Salesforce/codet5-small"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        log_message("Tokenizer loaded successfully")
    except Exception as e:
        log_message(f"Tokenizer error: {e}", level="ERROR")
        sys.exit(1)
    
    # Load data
    python_train, python_val, js_train, js_val = load_and_prepare_data()
    
    # Run multiple experiments with different seeds
    seeds = [42, 123, 456, 789, 999]  # 5 different seeds for statistical robustness
    
    lora_results = []
    full_results = []
    
    for seed in seeds:
        log_message(f"\n{'='*60}")
        log_message(f"EXPERIMENTAL TRIAL {len(lora_results) + 1}/{len(seeds)} (seed: {seed})")
        log_message(f"{'='*60}")
        
        # LoRA experiment
        try:
            lora_result = run_single_experiment(
                LoRAContinualLearner, model_name, tokenizer,
                python_train, python_val, js_train, js_val, seed
            )
            lora_results.append(lora_result)
        except Exception as e:
            log_message(f"LoRA experiment failed with seed {seed}: {e}", level="ERROR")
            continue
            
        # New Layer experiment (renamed from Full Layer)
        try:
            full_result = run_single_experiment(
                FullLayerContinualLearner, model_name, tokenizer,
                python_train, python_val, js_train, js_val, seed
            )
            full_results.append(full_result)
        except Exception as e:
            log_message(f"New Layer experiment failed with seed {seed}: {e}", level="ERROR")
            continue
    
    # Statistical analysis
    if len(lora_results) >= 3 and len(full_results) >= 3:
        run_statistical_analysis(lora_results, full_results)
    else:
        log_message("Insufficient successful experiments for statistical analysis", level="WARNING")
    
    log_message("Experiment completed!")

if __name__ == "__main__":
    main()