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

# Add utils to path for model evaluator, data loader, and device manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_evaluator import ModelEvaluator, ContinualLearningEvaluator
from utils.data_loader import load_and_prepare_data
from utils.device_manager import DeviceManager

@dataclass
class ExperimentResults:
    """Store results from a single experimental run with comprehensive metrics"""
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
            'forgetting_rate': self.forgetting_rate
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

def add_trainable_transformer_layer(model):
    """Add a new trainable transformer layer by creating a new model with extended architecture"""
    from copy import deepcopy
    import torch.nn as nn
    
    try:
        # Instead of modifying the existing model, create a new one with extended architecture
        original_config = model.config
        
        # Create new config with one additional layer
        new_config = deepcopy(original_config)
        new_config.num_layers = original_config.num_layers + 1
        
        # Create new model with extended architecture
        new_model = T5ForConditionalGeneration(new_config).to(model.device)
        
        # Copy weights from original model to new model (except the new layer)
        with torch.no_grad():
            # Copy encoder layers
            for i in range(original_config.num_layers):
                new_model.encoder.block[i].load_state_dict(model.encoder.block[i].state_dict())
            
            # Copy decoder layers
            for i in range(original_config.num_decoder_layers):
                new_model.decoder.block[i].load_state_dict(model.decoder.block[i].state_dict())
            
            # Copy other components
            new_model.shared.load_state_dict(model.shared.state_dict())
            new_model.encoder.final_layer_norm.load_state_dict(model.encoder.final_layer_norm.state_dict())
            new_model.decoder.final_layer_norm.load_state_dict(model.decoder.final_layer_norm.state_dict())
            new_model.lm_head.load_state_dict(model.lm_head.state_dict())
        
        # Freeze all copied parameters
        for param in new_model.parameters():
            param.requires_grad = False
        
        # Only make the new encoder layer trainable
        new_layer_idx = original_config.num_layers  # The new layer index
        for param in new_model.encoder.block[new_layer_idx].parameters():
            param.requires_grad = True
            # Initialize with small random values
            param.data = param.data * 0.01
        
        trainable_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in new_model.parameters())
        
        log_message(f"Created new model with additional layer: {trainable_params:,} trainable / {total_params:,} total parameters")
        
        return new_model
        
    except Exception as e:
        log_message(f"Error creating extended model: {e}", level="ERROR")
        raise

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
            torch_dtype=device_manager.torch_dtype
        ).to(self.device)
        self.base_model = device_manager.optimize_for_device(self.base_model)
        
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
        
        # Validation: Ensure model is properly loaded
        if self.current_model is None:
            raise RuntimeError(f"Failed to load LoRA adapter for task {task_name}")
        
        log_message(f"Successfully switched to LoRA adapter for {task_name} (device: {self.current_model.device})")
        
    def evaluate_task(self, eval_data, task_name: str, num_samples: int = 500) -> Tuple[float, float]:
        """Evaluate specific task using its adapter"""
        self.switch_to_task(task_name)
        log_message(f"Evaluating {task_name} task with LoRA adapter (current_task: {self.current_task})")
        
        # Infer language from task name
        language = "python" if "python" in task_name.lower() else "javascript" if "javascript" in task_name.lower() else None
        return self._evaluate_model(self.current_model, eval_data, num_samples, language)
        
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
        
    def _evaluate_model(self, model, data, num_samples: int, language: str = None) -> Tuple[float, float]:
        """Internal evaluation method using ModelEvaluator"""
        evaluator = ModelEvaluator(self.tokenizer)
        results = evaluator.evaluate_basic(model, data, language, num_samples)
        return results

class FullLayerContinualLearner(ContinualLearner):
    """New layer training with task-specific layer swapping (fair comparison with LoRA)"""
    
    def __init__(self, model_name: str, tokenizer, device: str):
        super().__init__(model_name, tokenizer, device)
        self.checkpoints = {}
        self.current_task = None
        
    def prepare_model(self) -> None:
        """Initialize the base model and freeze its weights"""
        super().prepare_model()
        # Freeze all base model parameters
        freeze_base_model(self.base_model)
        log_message(f"Base model prepared with frozen weights")
        
    def train_task(self, train_data, task_name: str, epochs: int = 5, batch_size: int = 16) -> float:
        """Train new transformer layer for specific task (independent, not cumulative)"""
        log_message(f"Training new transformer layer for {task_name}...")
        
        # Always start with clean base model (like LoRA starts fresh each time)
        model = deepcopy(self.base_model)
            
        # Add and configure new trainable layer for this specific task
        model = add_trainable_transformer_layer(model)
        
        # Verify only new layer is trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        log_message(f"Training {trainable_params:,} / {total_params:,} parameters ({100*trainable_params/total_params:.2f}%)")
        
        # Train only the new layer
        training_time = self._train_model(model, train_data, epochs, batch_size)
        
        # Save task-specific checkpoint (base + this task's layer only)
        checkpoint_path = f"checkpoints/{task_name}"
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        self.checkpoints[task_name] = checkpoint_path
        
        log_message(f"Task-specific checkpoint for {task_name} saved to {checkpoint_path}")
        log_message(f"Model architecture: Base (frozen) + {task_name} Layer (trained)")
        return training_time
        
    def switch_to_task(self, task_name: str) -> None:
        """Switch to specific task checkpoint (base + task-specific layer only)"""
        if task_name not in self.checkpoints:
            raise ValueError(f"No checkpoint found for task {task_name}")
            
        self.current_model = T5ForConditionalGeneration.from_pretrained(
            self.checkpoints[task_name]
        ).to(self.device)
        self.current_task = task_name
        
        # Validation: Ensure model is properly loaded
        if self.current_model is None:
            raise RuntimeError(f"Failed to load checkpoint for task {task_name}")
            
        log_message(f"Successfully switched to checkpoint for {task_name} (device: {self.current_model.device})")
        
    def evaluate_task(self, eval_data, task_name: str, num_samples: int = 500) -> Tuple[float, float]:
        """Evaluate specific task using its own checkpoint (task-specific layer)"""
        self.switch_to_task(task_name)
        log_message(f"Evaluating {task_name} task with new layer (current_task: {self.current_task})")
        
        # Infer language from task name
        language = "python" if "python" in task_name.lower() else "javascript" if "javascript" in task_name.lower() else None
        return self._evaluate_model(self.current_model, eval_data, num_samples, language)
        
    def _train_model(self, model, data, epochs: int, batch_size: int) -> float:
        """Internal training method"""
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
        
    def _evaluate_model(self, model, data, num_samples: int, language: str = None) -> Tuple[float, float]:
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
                    ).to(model.device)
                    
                    outputs = model.generate(
                        **inputs, 
                        max_length=512, 
                        num_beams=3,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        temperature=0.7
                    )
                    
                    pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Validate generated text
                    if not pred_text or not pred_text.strip():
                        # Empty generation
                        bleu_scores.append(0.0)
                        meteor_scores.append(0.0)
                        edit_distances.append(1.0)
                        ast_similarities.append(0.0)
                        complexity_scores.append(1.0)
                        pass_scores.append(0.0)
                        continue
                    
                    # Calculate BLEU score
                    target_tokens = self.tokenizer.tokenize(target_text)
                    pred_tokens = self.tokenizer.tokenize(pred_text)
                    
                    if target_tokens and pred_tokens:
                        bleu = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothing)
                        bleu_scores.append(bleu)
                    else:
                        bleu_scores.append(0.0)
                        
                    # Test syntactic correctness - use language parameter from dataset or infer
                    try:
                        # Auto-detect language if not provided
                        detected_language = language or data[i].get("language", "").lower()
                        
                        if detected_language == "python" or any(keyword in source_code.lower() for keyword in ['def ', 'import ', 'class ', 'print(']):
                            # Python code - try to compile
                            compile(pred_text, "<string>", "exec")
                            pass_scores.append(1.0)
                        elif detected_language == "javascript" or any(keyword in source_code.lower() for keyword in ['function ', 'var ', 'let ', 'const ', 'console.']):
                            # JavaScript code - basic syntax check
                            if pred_text.strip() and '{' in pred_text and '}' in pred_text and not pred_text.strip().startswith('//'):
                                pass_scores.append(1.0)
                            else:
                                pass_scores.append(0.0)
                        else:
                            # Unknown language - basic non-empty check
                            pass_scores.append(1.0 if pred_text.strip() else 0.0)
                    except:
                        pass_scores.append(0.0)
                        
                except Exception as e:
                    bleu_scores.append(0.0)
                    pass_scores.append(0.0)
                    
        return np.mean(bleu_scores) if bleu_scores else 0.0, np.mean(pass_scores) if pass_scores else 0.0

def get_memory_usage():
    """Get current memory usage in GB"""
    return psutil.Process().memory_info().rss / 1024**3

def calculate_edit_distance(pred_text: str, target_text: str) -> float:
    """Calculate normalized edit distance between two code strings"""
    if not pred_text or not target_text:
        return 1.0
    
    # Normalize whitespace and remove comments for fair comparison
    pred_clean = re.sub(r'\s+', ' ', pred_text.strip())
    target_clean = re.sub(r'\s+', ' ', target_text.strip())
    
    # Calculate Levenshtein distance
    distance = difflib.SequenceMatcher(None, pred_clean, target_clean).ratio()
    return 1.0 - distance  # Convert similarity to distance

def calculate_ast_similarity(pred_text: str, target_text: str, language: str = "python") -> float:
    """Calculate AST similarity between generated and target code"""
    try:
        if language.lower() == "python":
            # Parse both codes into ASTs
            pred_ast = ast.parse(pred_text)
            target_ast = ast.parse(target_text)
            
            # Convert ASTs to comparable structures
            pred_nodes = [type(node).__name__ for node in ast.walk(pred_ast)]
            target_nodes = [type(node).__name__ for node in ast.walk(target_ast)]
            
            # Calculate structural similarity
            if not pred_nodes or not target_nodes:
                return 0.0
                
            common_nodes = len(set(pred_nodes) & set(target_nodes))
            total_nodes = len(set(pred_nodes) | set(target_nodes))
            
            return common_nodes / total_nodes if total_nodes > 0 else 0.0
        else:
            # For JavaScript, use a simpler token-based approach
            pred_tokens = re.findall(r'\w+|[{}();,.]', pred_text)
            target_tokens = re.findall(r'\w+|[{}();,.]', target_text)
            
            if not pred_tokens or not target_tokens:
                return 0.0
                
            common_tokens = len(set(pred_tokens) & set(target_tokens))
            total_tokens = len(set(pred_tokens) | set(target_tokens))
            
            return common_tokens / total_tokens if total_tokens > 0 else 0.0
            
    except (SyntaxError, ValueError):
        return 0.0

def calculate_code_complexity(code_text: str, language: str = "python") -> float:
    """Calculate cyclomatic complexity of code"""
    try:
        if language.lower() == "python":
            # Count control flow statements for Python
            control_statements = len(re.findall(r'\b(if|elif|else|for|while|try|except|finally|with|def|class)\b', code_text))
            logical_operators = len(re.findall(r'\b(and|or)\b', code_text))
            return control_statements + logical_operators + 1
        else:
            # Count control flow statements for JavaScript
            control_statements = len(re.findall(r'\b(if|else|for|while|do|switch|case|try|catch|finally|function)\b', code_text))
            logical_operators = len(re.findall(r'(\&\&|\|\|)', code_text))
            return control_statements + logical_operators + 1
    except:
        return 1.0

def calculate_meteor_score_safe(pred_text: str, target_text: str) -> float:
    """Calculate METEOR score with error handling"""
    try:
        if not pred_text.strip() or not target_text.strip():
            return 0.0
            
        # Tokenize for METEOR
        pred_tokens = pred_text.split()
        target_tokens = target_text.split()
        
        if not pred_tokens or not target_tokens:
            return 0.0
            
        # Calculate METEOR score
        score = meteor_score([target_tokens], pred_tokens)
        return score
    except:
        return 0.0

class ComprehensiveEvaluator:
    """Comprehensive evaluation with multiple metrics using ModelEvaluator"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.evaluator = ModelEvaluator(tokenizer)
        
    def evaluate_comprehensive(self, model, data, language: str, num_samples: int = 100) -> Dict[str, float]:
        """Run comprehensive evaluation with all metrics using ModelEvaluator"""
        results = self.evaluator.evaluate_comprehensive(model, data, language, num_samples)
        return results.to_dict()

def calculate_continual_learning_metrics(baseline_python: Dict, baseline_js: Dict, 
                                       python_after_python: Dict, js_after_python: Dict,
                                       python_after_js: Dict, js_after_js: Dict) -> Dict[str, float]:
    """Calculate continual learning specific metrics using ContinualLearningEvaluator"""
    
    # Convert dict results to EvaluationResults objects for ContinualLearningEvaluator
    from utils.model_evaluator import EvaluationResults
    
    baseline_task1 = EvaluationResults(
        bleu=baseline_python['bleu'], meteor=baseline_python.get('meteor', 0.0),
        pass_rate=baseline_python.get('pass_rate', 0.0), edit_distance=baseline_python.get('edit_distance', 1.0),
        ast_similarity=baseline_python.get('ast_similarity', 0.0), complexity=baseline_python.get('complexity', 1.0),
        num_samples=baseline_python.get('num_samples', 0), language="python"
    )
    
    baseline_task2 = EvaluationResults(
        bleu=baseline_js['bleu'], meteor=baseline_js.get('meteor', 0.0),
        pass_rate=baseline_js.get('pass_rate', 0.0), edit_distance=baseline_js.get('edit_distance', 1.0),
        ast_similarity=baseline_js.get('ast_similarity', 0.0), complexity=baseline_js.get('complexity', 1.0),
        num_samples=baseline_js.get('num_samples', 0), language="javascript"
    )
    
    task1_after_task1 = EvaluationResults(
        bleu=python_after_python['bleu'], meteor=python_after_python.get('meteor', 0.0),
        pass_rate=python_after_python.get('pass_rate', 0.0), edit_distance=python_after_python.get('edit_distance', 1.0),
        ast_similarity=python_after_python.get('ast_similarity', 0.0), complexity=python_after_python.get('complexity', 1.0),
        num_samples=python_after_python.get('num_samples', 0), language="python"
    )
    
    task2_after_task1 = EvaluationResults(
        bleu=js_after_python['bleu'], meteor=js_after_python.get('meteor', 0.0),
        pass_rate=js_after_python.get('pass_rate', 0.0), edit_distance=js_after_python.get('edit_distance', 1.0),
        ast_similarity=js_after_python.get('ast_similarity', 0.0), complexity=js_after_python.get('complexity', 1.0),
        num_samples=js_after_python.get('num_samples', 0), language="javascript"
    )
    
    task1_after_task2 = EvaluationResults(
        bleu=python_after_js['bleu'], meteor=python_after_js.get('meteor', 0.0),
        pass_rate=python_after_js.get('pass_rate', 0.0), edit_distance=python_after_js.get('edit_distance', 1.0),
        ast_similarity=python_after_js.get('ast_similarity', 0.0), complexity=python_after_js.get('complexity', 1.0),
        num_samples=python_after_js.get('num_samples', 0), language="python"
    )
    
    task2_after_task2 = EvaluationResults(
        bleu=js_after_js['bleu'], meteor=js_after_js.get('meteor', 0.0),
        pass_rate=js_after_js.get('pass_rate', 0.0), edit_distance=js_after_js.get('edit_distance', 1.0),
        ast_similarity=js_after_js.get('ast_similarity', 0.0), complexity=js_after_js.get('complexity', 1.0),
        num_samples=js_after_js.get('num_samples', 0), language="javascript"
    )
    
    # Use ContinualLearningEvaluator for consistent metrics calculation
    dummy_evaluator = ModelEvaluator()  # No tokenizer needed for metrics calculation
    cl_evaluator = ContinualLearningEvaluator(dummy_evaluator)
    
    return cl_evaluator.calculate_continual_learning_metrics(
        baseline_task1, baseline_task2,
        task1_after_task1, task2_after_task1,
        task1_after_task2, task2_after_task2
    )

def run_single_experiment(learner_class, model_name: str, tokenizer, python_train, python_val, js_train, js_val, seed: int) -> ExperimentResults:
    """Run a single experimental trial with comprehensive evaluation metrics"""
    set_seed(seed)
    log_message(f"Running {learner_class.__name__} experiment with seed {seed}")
    
    # Initialize learner and comprehensive evaluator
    learner = learner_class(model_name, tokenizer, device)
    learner.prepare_model()
    evaluator = ComprehensiveEvaluator(tokenizer)
    
    start_memory = get_memory_usage()
    
    # === BASELINE EVALUATION (before any training) ===
    log_message("=== BASELINE EVALUATION (Untrained Model) ===")
    baseline_python = evaluator.evaluate_comprehensive(learner.base_model, python_val, "python", 50)
    baseline_js = evaluator.evaluate_comprehensive(learner.base_model, js_val, "javascript", 50)
    
    log_message(f"Baseline Python - BLEU: {baseline_python['bleu']:.4f}, METEOR: {baseline_python['meteor']:.4f}, AST: {baseline_python['ast_similarity']:.4f}")
    log_message(f"Baseline JavaScript - BLEU: {baseline_js['bleu']:.4f}, METEOR: {baseline_js['meteor']:.4f}, AST: {baseline_js['ast_similarity']:.4f}")
    
    # === STEP 1: TRAIN ON PYTHON ===
    log_message("\n=== STEP 1: TRAINING ON PYTHON ===")
    python_training_time = learner.train_task(python_train, "python", epochs=2, batch_size=8)
    
    # Comprehensive evaluation after Python training
    log_message("Comprehensive evaluation after Python training...")
    
    # Properly load and evaluate the Python adapter
    learner.switch_to_task("python")
    log_message(f"Current model for Python evaluation: {type(learner.current_model).__name__}")
    python_after_python = evaluator.evaluate_comprehensive(learner.current_model, python_val, "python", 50)
    
    # For JavaScript, use base model since no JS-specific component exists yet
    js_after_python = baseline_js  # No change expected
    
    log_message(f"After Python training:")
    log_message(f"  Python BLEU: {baseline_python['bleu']:.4f} → {python_after_python['bleu']:.4f} (Δ{python_after_python['bleu']-baseline_python['bleu']:+.4f})")
    log_message(f"  Python METEOR: {baseline_python['meteor']:.4f} → {python_after_python['meteor']:.4f} (Δ{python_after_python['meteor']-baseline_python['meteor']:+.4f})")
    log_message(f"  Python AST Similarity: {baseline_python['ast_similarity']:.4f} → {python_after_python['ast_similarity']:.4f}")
    log_message(f"  Python Code Complexity: {python_after_python['complexity']:.2f}")
    
    # === STEP 2: TRAIN ON JAVASCRIPT ===
    log_message("\n=== STEP 2: TRAINING ON JAVASCRIPT ===")
    js_training_time = learner.train_task(js_train, "javascript", epochs=2, batch_size=8)
    
    # Comprehensive evaluation after JavaScript training
    log_message("Comprehensive evaluation after JavaScript training...")
    
    # Evaluate JavaScript with its own component
    learner.switch_to_task("javascript")
    log_message(f"Current model for JavaScript evaluation: {type(learner.current_model).__name__}")
    js_after_js = evaluator.evaluate_comprehensive(learner.current_model, js_val, "javascript", 50)
    
    # Re-evaluate Python to measure forgetting
    learner.switch_to_task("python") 
    log_message(f"Current model for Python re-evaluation: {type(learner.current_model).__name__}")
    python_after_js = evaluator.evaluate_comprehensive(learner.current_model, python_val, "python", 50)
    
    log_message(f"After JavaScript training:")
    log_message(f"  JavaScript BLEU: {baseline_js['bleu']:.4f} → {js_after_js['bleu']:.4f} (Δ{js_after_js['bleu']-baseline_js['bleu']:+.4f})")
    log_message(f"  JavaScript METEOR: {baseline_js['meteor']:.4f} → {js_after_js['meteor']:.4f}")
    log_message(f"  JavaScript AST Similarity: {js_after_js['ast_similarity']:.4f}")
    log_message(f"  Python BLEU: {python_after_python['bleu']:.4f} → {python_after_js['bleu']:.4f} (Δ{python_after_js['bleu']-python_after_python['bleu']:+.4f})")
    log_message(f"  Python Forgetting: {((python_after_python['bleu'] - python_after_js['bleu'])/python_after_python['bleu']*100):.1f}%")
    
    # === CONTINUAL LEARNING ANALYSIS ===
    cl_metrics = calculate_continual_learning_metrics(
        baseline_python, baseline_js, python_after_python, js_after_python, python_after_js, js_after_js
    )
    
    log_message(f"\n=== CONTINUAL LEARNING ANALYSIS ===")
    log_message(f"🔄 Forward Transfer: {cl_metrics['forward_transfer']:.4f}")
    log_message(f"🔙 Backward Interference: {cl_metrics['backward_interference']:.4f}")
    log_message(f"🧠 Retention Score: {cl_metrics['retention_score']:.4f}")
    
    # === SUMMARY ANALYSIS ===
    end_memory = get_memory_usage()
    total_training_time = python_training_time + js_training_time
    forgetting_rate = (python_after_python['bleu'] - python_after_js['bleu']) / python_after_python['bleu'] if python_after_python['bleu'] > 0 else 0
    
    log_message(f"\n=== COMPREHENSIVE SUMMARY ===")
    log_message(f"📊 Final Python Performance: BLEU {python_after_js['bleu']:.4f}, METEOR {python_after_js['meteor']:.4f}")
    log_message(f"📊 Final JavaScript Performance: BLEU {js_after_js['bleu']:.4f}, METEOR {js_after_js['meteor']:.4f}")
    log_message(f"⚡ Training Efficiency: {total_training_time:.2f} min, {end_memory - start_memory:.2f} GB")
    
    results = ExperimentResults(
        # Basic metrics
        python_bleu_before=python_after_python['bleu'],
        python_bleu_after=python_after_js['bleu'],
        js_bleu=js_after_js['bleu'],
        python_pass_before=python_after_python['pass_rate'],
        python_pass_after=python_after_js['pass_rate'],
        js_pass=js_after_js['pass_rate'],
        
        # Advanced semantic metrics  
        python_meteor_before=python_after_python['meteor'],
        python_meteor_after=python_after_js['meteor'],
        js_meteor=js_after_js['meteor'],
        python_edit_distance_before=python_after_python['edit_distance'],
        python_edit_distance_after=python_after_js['edit_distance'],
        js_edit_distance=js_after_js['edit_distance'],
        
        # Code quality metrics
        python_complexity_before=python_after_python['complexity'],
        python_complexity_after=python_after_js['complexity'],
        js_complexity=js_after_js['complexity'],
        python_ast_similarity_before=python_after_python['ast_similarity'],
        python_ast_similarity_after=python_after_js['ast_similarity'],
        js_ast_similarity=js_after_js['ast_similarity'],
        
        # Continual learning metrics
        forward_transfer=cl_metrics['forward_transfer'],
        backward_interference=cl_metrics['backward_interference'],
        retention_score=cl_metrics['retention_score'],
        
        # Efficiency metrics
        training_time=total_training_time,
        memory_usage=end_memory - start_memory,
        forgetting_rate=forgetting_rate
    )
    
    log_message(f"\n{learner_class.__name__} Comprehensive Results (seed {seed}):")
    log_message(f"  📊 Performance: Python BLEU {python_after_js['bleu']:.4f}, JS BLEU {js_after_js['bleu']:.4f}")
    log_message(f"  🧠 Continual Learning: Forgetting {forgetting_rate:.2%}, Retention {cl_metrics['retention_score']:.4f}")
    log_message(f"  ⚡ Efficiency: {total_training_time:.2f} min, {end_memory - start_memory:.2f} GB")
    
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
    log_message("FAIR COMPARISON: Both approaches use task-specific component swapping")
    log_message("LoRA: Frozen base + swappable adapter matrices (~0.1% parameters)")
    log_message("New Layer: Frozen base + swappable transformer layers (~5% parameters)")
    log_message("Both: Perfect task isolation, no cross-task interference during training")
    
    # Set seed for reproducibility
    device_manager.set_seed(42)
    
    # Initialize tokenizer
    model_name = "Salesforce/codet5-small"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        log_message("Tokenizer loaded successfully")
    except Exception as e:
        log_message(f"Tokenizer error: {e}", level="ERROR")
        sys.exit(1)
    
    # Load data using the new unified data loader (same splits as original)
    python_train, python_val, js_train, js_val = load_and_prepare_data(
        python_train_size=15000,
        python_val_size=5000,
        js_train_size=15000,
        js_val_size=5000,
        format_type="huggingface",
        seed=42
    )
    
    # Run multiple experiments with different seeds
    seeds = [42]  # Start with just one seed for testing
    
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