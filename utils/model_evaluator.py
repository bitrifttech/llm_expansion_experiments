"""
Model Evaluation Utilities

This module provides comprehensive evaluation capabilities for code generation models,
consolidating evaluation logic from various experiments into reusable components.

Key Features:
- Multiple evaluation metrics (BLEU, METEOR, AST similarity, etc.)
- Language-specific syntax validation
- Code complexity analysis
- Continual learning metrics
- Configurable evaluation parameters
- Consistent scoring across experiments

Usage:
    from utils.model_evaluator import ModelEvaluator, EvaluationConfig
    
    evaluator = ModelEvaluator(tokenizer, config=EvaluationConfig())
    results = evaluator.evaluate_comprehensive(model, data, language="python")
"""

import torch
import numpy as np
import ast
import re
import difflib
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import warnings

# Suppress NLTK warnings
warnings.filterwarnings("ignore", category=UserWarning, module='nltk')


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    # Generation parameters
    max_length: int = 512
    num_beams: int = 3
    no_repeat_ngram_size: int = 2
    do_sample: bool = True
    temperature: float = 0.7
    
    # Evaluation parameters
    default_num_samples: int = 100
    input_fraction: float = 0.5  # Fraction of source code to use as input
    
    # Metric weights (for composite scoring)
    bleu_weight: float = 0.3
    meteor_weight: float = 0.2
    ast_similarity_weight: float = 0.2
    pass_rate_weight: float = 0.2
    complexity_weight: float = 0.1
    
    # Language detection keywords
    python_keywords: List[str] = field(default_factory=lambda: [
        'def ', 'import ', 'class ', 'print(', 'if __name__', 'return ', 'lambda '
    ])
    javascript_keywords: List[str] = field(default_factory=lambda: [
        'function ', 'var ', 'let ', 'const ', 'console.', '=>', 'require('
    ])


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results"""
    # Core metrics
    bleu: float
    meteor: float
    pass_rate: float
    
    # Code quality metrics
    edit_distance: float
    ast_similarity: float
    complexity: float
    
    # Meta information
    num_samples: int
    language: str
    
    # Individual scores (for debugging/analysis)
    individual_bleu_scores: List[float] = field(default_factory=list)
    individual_meteor_scores: List[float] = field(default_factory=list)
    individual_pass_scores: List[float] = field(default_factory=list)
    
    def composite_score(self, config: EvaluationConfig) -> float:
        """Calculate weighted composite score"""
        return (
            self.bleu * config.bleu_weight +
            self.meteor * config.meteor_weight +
            self.ast_similarity * config.ast_similarity_weight +
            self.pass_rate * config.pass_rate_weight +
            (1.0 - self.complexity / 10.0) * config.complexity_weight  # Normalize complexity
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'bleu': self.bleu,
            'meteor': self.meteor,
            'pass_rate': self.pass_rate,
            'edit_distance': self.edit_distance,
            'ast_similarity': self.ast_similarity,
            'complexity': self.complexity,
            'num_samples': self.num_samples,
            'language': self.language
        }


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics"""
    
    def __init__(self, tokenizer=None, config: Optional[EvaluationConfig] = None):
        """
        Initialize the evaluator
        
        Args:
            tokenizer: HuggingFace tokenizer for the model (optional, required only for generation)
            config: Evaluation configuration (uses defaults if None)
        """
        self.tokenizer = tokenizer
        self.config = config or EvaluationConfig()
        self.smoothing = SmoothingFunction().method1
        
    def evaluate_comprehensive(
        self, 
        model, 
        data, 
        language: Optional[str] = None, 
        num_samples: Optional[int] = None
    ) -> EvaluationResults:
        """
        Run comprehensive evaluation with all metrics
        
        Args:
            model: The model to evaluate
            data: Dataset to evaluate on
            language: Programming language ("python", "javascript", or None for auto-detect)
            num_samples: Number of samples to evaluate (uses config default if None)
            
        Returns:
            EvaluationResults object with all metrics
        """
        model.eval()
        num_samples = num_samples or self.config.default_num_samples
        eval_samples = min(num_samples, len(data))
        
        # Metric collectors
        bleu_scores = []
        meteor_scores = []
        edit_distances = []
        ast_similarities = []
        complexity_scores = []
        pass_scores = []
        
        with torch.no_grad():
            for i in range(eval_samples):
                try:
                    # Extract source code
                    source_code = self._extract_source_code(data[i])
                    if not source_code:
                        self._append_zero_scores(
                            bleu_scores, meteor_scores, edit_distances,
                            ast_similarities, complexity_scores, pass_scores
                        )
                        continue
                    
                    # Detect language if not provided
                    detected_language = language or self._detect_language(source_code)
                    
                    # Prepare input and target
                    input_text, target_text = self._prepare_input_target(source_code)
                    
                    # Generate prediction
                    pred_text = self._generate_prediction(model, input_text)
                    
                    if not pred_text or not pred_text.strip():
                        self._append_zero_scores(
                            bleu_scores, meteor_scores, edit_distances,
                            ast_similarities, complexity_scores, pass_scores
                        )
                        continue
                    
                    # Calculate all metrics
                    bleu_score = self._calculate_bleu(pred_text, target_text)
                    meteor_score_val = self._calculate_meteor(pred_text, target_text)
                    edit_distance = self._calculate_edit_distance(pred_text, target_text)
                    ast_similarity = self._calculate_ast_similarity(pred_text, target_text, detected_language)
                    complexity = self._calculate_complexity(pred_text, detected_language)
                    pass_score = self._calculate_pass_rate(pred_text, source_code, detected_language)
                    
                    # Append scores
                    bleu_scores.append(bleu_score)
                    meteor_scores.append(meteor_score_val)
                    edit_distances.append(edit_distance)
                    ast_similarities.append(ast_similarity)
                    complexity_scores.append(complexity)
                    pass_scores.append(pass_score)
                    
                except Exception as e:
                    # Log error and append zero scores
                    print(f"Warning: Evaluation error for sample {i}: {e}")
                    self._append_zero_scores(
                        bleu_scores, meteor_scores, edit_distances,
                        ast_similarities, complexity_scores, pass_scores
                    )
        
        # Calculate final metrics
        return EvaluationResults(
            bleu=np.mean(bleu_scores) if bleu_scores else 0.0,
            meteor=np.mean(meteor_scores) if meteor_scores else 0.0,
            pass_rate=np.mean(pass_scores) if pass_scores else 0.0,
            edit_distance=np.mean(edit_distances) if edit_distances else 1.0,
            ast_similarity=np.mean(ast_similarities) if ast_similarities else 0.0,
            complexity=np.mean(complexity_scores) if complexity_scores else 1.0,
            num_samples=len(bleu_scores),
            language=language or "auto-detected",
            individual_bleu_scores=bleu_scores,
            individual_meteor_scores=meteor_scores,
            individual_pass_scores=pass_scores
        )
    
    def evaluate_basic(
        self, 
        model, 
        data, 
        language: Optional[str] = None, 
        num_samples: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Basic evaluation returning only BLEU and pass rate (for backward compatibility)
        
        Returns:
            Tuple of (bleu_score, pass_rate)
        """
        results = self.evaluate_comprehensive(model, data, language, num_samples)
        return results.bleu, results.pass_rate
    
    def _extract_source_code(self, data_item) -> Optional[str]:
        """Extract source code from data item (handles different data formats)"""
        if isinstance(data_item, dict):
            # Try common field names
            for field in ['func_code_string', 'code', 'source_code', 'text']:
                if field in data_item and data_item[field]:
                    return str(data_item[field]).strip()
        elif isinstance(data_item, str):
            return data_item.strip()
        elif hasattr(data_item, 'func_code_string'):
            return str(data_item.func_code_string).strip()
        elif hasattr(data_item, 'code'):
            return str(data_item.code).strip()
        
        return None
    
    def _detect_language(self, source_code: str) -> str:
        """Auto-detect programming language from source code"""
        source_lower = source_code.lower()
        
        # Check for Python keywords
        python_matches = sum(1 for keyword in self.config.python_keywords 
                           if keyword in source_lower)
        
        # Check for JavaScript keywords
        js_matches = sum(1 for keyword in self.config.javascript_keywords 
                        if keyword in source_lower)
        
        if python_matches > js_matches:
            return "python"
        elif js_matches > python_matches:
            return "javascript"
        else:
            return "unknown"
    
    def _prepare_input_target(self, source_code: str) -> Tuple[str, str]:
        """Prepare input and target text from source code"""
        split_point = int(len(source_code) * self.config.input_fraction)
        input_text = source_code[:split_point]
        target_text = source_code
        return input_text, target_text
    
    def _generate_prediction(self, model, input_text: str) -> str:
        """Generate prediction from model"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for model generation. Please provide a tokenizer when initializing ModelEvaluator.")
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _calculate_bleu(self, predicted: str, target: str) -> float:
        """Calculate BLEU score"""
        try:
            if self.tokenizer is not None:
                # Use tokenizer if available
                target_tokens = self.tokenizer.tokenize(target)
                pred_tokens = self.tokenizer.tokenize(predicted)
            else:
                # Fallback to simple word tokenization
                target_tokens = target.split()
                pred_tokens = predicted.split()
            
            if not target_tokens or not pred_tokens:
                return 0.0
                
            return sentence_bleu([target_tokens], pred_tokens, smoothing_function=self.smoothing)
        except Exception:
            return 0.0
    
    def _calculate_meteor(self, predicted: str, target: str) -> float:
        """Calculate METEOR score with error handling"""
        try:
            if not predicted.strip() or not target.strip():
                return 0.0
                
            pred_tokens = predicted.split()
            target_tokens = target.split()
            
            if not pred_tokens or not target_tokens:
                return 0.0
                
            return meteor_score([target_tokens], pred_tokens)
        except Exception:
            return 0.0
    
    def _calculate_edit_distance(self, predicted: str, target: str) -> float:
        """Calculate normalized edit distance"""
        try:
            if not predicted or not target:
                return 1.0
            
            # Normalize whitespace for fair comparison
            pred_clean = re.sub(r'\s+', ' ', predicted.strip())
            target_clean = re.sub(r'\s+', ' ', target.strip())
            
            # Calculate similarity ratio and convert to distance
            similarity = difflib.SequenceMatcher(None, pred_clean, target_clean).ratio()
            return 1.0 - similarity
        except Exception:
            return 1.0
    
    def _calculate_ast_similarity(self, predicted: str, target: str, language: str) -> float:
        """Calculate AST similarity between predicted and target code"""
        try:
            if language.lower() == "python":
                return self._calculate_python_ast_similarity(predicted, target)
            elif language.lower() == "javascript":
                return self._calculate_javascript_ast_similarity(predicted, target)
            else:
                # Fallback to token-based similarity
                return self._calculate_token_similarity(predicted, target)
        except Exception:
            return 0.0
    
    def _calculate_python_ast_similarity(self, predicted: str, target: str) -> float:
        """Calculate Python AST similarity"""
        try:
            pred_ast = ast.parse(predicted)
            target_ast = ast.parse(target)
            
            # Extract node types
            pred_nodes = [type(node).__name__ for node in ast.walk(pred_ast)]
            target_nodes = [type(node).__name__ for node in ast.walk(target_ast)]
            
            if not pred_nodes or not target_nodes:
                return 0.0
            
            # Calculate Jaccard similarity
            pred_set = set(pred_nodes)
            target_set = set(target_nodes)
            intersection = len(pred_set & target_set)
            union = len(pred_set | target_set)
            
            return intersection / union if union > 0 else 0.0
        except (SyntaxError, ValueError):
            return 0.0
    
    def _calculate_javascript_ast_similarity(self, predicted: str, target: str) -> float:
        """Calculate JavaScript AST similarity (simplified token-based approach)"""
        try:
            # Extract JavaScript tokens/keywords
            pred_tokens = re.findall(r'\w+|[{}();,.]', predicted)
            target_tokens = re.findall(r'\w+|[{}();,.]', target)
            
            if not pred_tokens or not target_tokens:
                return 0.0
            
            # Calculate Jaccard similarity
            pred_set = set(pred_tokens)
            target_set = set(target_tokens)
            intersection = len(pred_set & target_set)
            union = len(pred_set | target_set)
            
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_token_similarity(self, predicted: str, target: str) -> float:
        """Fallback token-based similarity"""
        try:
            pred_tokens = set(predicted.split())
            target_tokens = set(target.split())
            
            if not pred_tokens or not target_tokens:
                return 0.0
            
            intersection = len(pred_tokens & target_tokens)
            union = len(pred_tokens | target_tokens)
            
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_complexity(self, code: str, language: str) -> float:
        """Calculate cyclomatic complexity"""
        try:
            if language.lower() == "python":
                # Python complexity
                control_statements = len(re.findall(
                    r'\b(if|elif|else|for|while|try|except|finally|with|def|class)\b', 
                    code
                ))
                logical_operators = len(re.findall(r'\b(and|or)\b', code))
                return control_statements + logical_operators + 1
            elif language.lower() == "javascript":
                # JavaScript complexity
                control_statements = len(re.findall(
                    r'\b(if|else|for|while|do|switch|case|try|catch|finally|function)\b', 
                    code
                ))
                logical_operators = len(re.findall(r'(\&\&|\|\|)', code))
                return control_statements + logical_operators + 1
            else:
                # Generic complexity based on lines and basic constructs
                lines = len(code.split('\n'))
                constructs = code.count('{') + code.count('(') + code.count('[')
                return lines + constructs
        except Exception:
            return 1.0
    
    def _calculate_pass_rate(self, predicted: str, source_code: str, language: str) -> float:
        """Calculate pass rate (syntactic correctness)"""
        try:
            if language.lower() == "python":
                # Try to compile Python code
                compile(predicted, "<string>", "exec")
                return 1.0
            elif language.lower() == "javascript":
                # Basic JavaScript syntax check
                if (predicted.strip() and 
                    '{' in predicted and '}' in predicted and 
                    not predicted.strip().startswith('//')):
                    return 1.0
                else:
                    return 0.0
            else:
                # Basic non-empty check for unknown languages
                return 1.0 if predicted.strip() else 0.0
        except Exception:
            return 0.0
    
    def _append_zero_scores(self, bleu_scores, meteor_scores, edit_distances, 
                          ast_similarities, complexity_scores, pass_scores):
        """Append zero scores for failed samples"""
        bleu_scores.append(0.0)
        meteor_scores.append(0.0)
        edit_distances.append(1.0)
        ast_similarities.append(0.0)
        complexity_scores.append(1.0)
        pass_scores.append(0.0)


class ContinualLearningEvaluator:
    """Specialized evaluator for continual learning metrics"""
    
    def __init__(self, base_evaluator: ModelEvaluator):
        """
        Initialize with a base evaluator
        
        Args:
            base_evaluator: ModelEvaluator instance to use for basic evaluation
        """
        self.evaluator = base_evaluator
    
    def calculate_continual_learning_metrics(
        self,
        baseline_task1: EvaluationResults,
        baseline_task2: EvaluationResults,
        task1_after_task1: EvaluationResults,
        task2_after_task1: EvaluationResults,
        task1_after_task2: EvaluationResults,
        task2_after_task2: EvaluationResults
    ) -> Dict[str, float]:
        """
        Calculate continual learning specific metrics
        
        Args:
            baseline_task1: Task 1 performance before any training
            baseline_task2: Task 2 performance before any training
            task1_after_task1: Task 1 performance after Task 1 training
            task2_after_task1: Task 2 performance after Task 1 training
            task1_after_task2: Task 1 performance after Task 2 training
            task2_after_task2: Task 2 performance after Task 2 training
            
        Returns:
            Dictionary with continual learning metrics
        """
        # Forward Transfer: How much Task 1 training helped Task 2
        forward_transfer = max(0, task2_after_task1.bleu - baseline_task2.bleu)
        
        # Backward Interference: How much Task 2 training hurt Task 1
        backward_interference = max(0, task1_after_task1.bleu - task1_after_task2.bleu)
        
        # Retention Score: Overall knowledge retention
        task1_retention = (task1_after_task2.bleu / task1_after_task1.bleu 
                          if task1_after_task1.bleu > 0 else 0)
        task2_improvement = (task2_after_task2.bleu / baseline_task2.bleu 
                           if baseline_task2.bleu > 0 else 0)
        retention_score = (task1_retention + task2_improvement) / 2
        
        # Forgetting Rate: Percentage of performance lost
        forgetting_rate = (backward_interference / task1_after_task1.bleu 
                          if task1_after_task1.bleu > 0 else 0)
        
        return {
            'forward_transfer': forward_transfer,
            'backward_interference': backward_interference,
            'retention_score': retention_score,
            'forgetting_rate': forgetting_rate,
            'task1_retention': task1_retention,
            'task2_improvement': task2_improvement
        }
    
    def evaluate_continual_learning_experiment(
        self,
        model_before_training,
        model_after_task1,
        model_after_task2,
        task1_data,
        task2_data,
        task1_language: str = "python",
        task2_language: str = "javascript",
        num_samples: int = 100
    ) -> Dict[str, Union[EvaluationResults, Dict[str, float]]]:
        """
        Complete continual learning evaluation
        
        Returns:
            Dictionary containing all evaluation results and continual learning metrics
        """
        # Baseline evaluations
        baseline_task1 = self.evaluator.evaluate_comprehensive(
            model_before_training, task1_data, task1_language, num_samples
        )
        baseline_task2 = self.evaluator.evaluate_comprehensive(
            model_before_training, task2_data, task2_language, num_samples
        )
        
        # After Task 1 training
        task1_after_task1 = self.evaluator.evaluate_comprehensive(
            model_after_task1, task1_data, task1_language, num_samples
        )
        task2_after_task1 = self.evaluator.evaluate_comprehensive(
            model_after_task1, task2_data, task2_language, num_samples
        )
        
        # After Task 2 training
        task1_after_task2 = self.evaluator.evaluate_comprehensive(
            model_after_task2, task1_data, task1_language, num_samples
        )
        task2_after_task2 = self.evaluator.evaluate_comprehensive(
            model_after_task2, task2_data, task2_language, num_samples
        )
        
        # Calculate continual learning metrics
        cl_metrics = self.calculate_continual_learning_metrics(
            baseline_task1, baseline_task2,
            task1_after_task1, task2_after_task1,
            task1_after_task2, task2_after_task2
        )
        
        return {
            'baseline_task1': baseline_task1,
            'baseline_task2': baseline_task2,
            'task1_after_task1': task1_after_task1,
            'task2_after_task1': task2_after_task1,
            'task1_after_task2': task1_after_task2,
            'task2_after_task2': task2_after_task2,
            'continual_learning_metrics': cl_metrics
        }


# Convenience functions for backward compatibility
def evaluate_model_basic(model, data, tokenizer, language: str = None, num_samples: int = 100) -> Tuple[float, float]:
    """
    Basic model evaluation function for backward compatibility
    
    Returns:
        Tuple of (bleu_score, pass_rate)
    """
    evaluator = ModelEvaluator(tokenizer)
    return evaluator.evaluate_basic(model, data, language, num_samples)


def evaluate_model_comprehensive(model, data, tokenizer, language: str = None, num_samples: int = 100) -> EvaluationResults:
    """
    Comprehensive model evaluation function
    
    Returns:
        EvaluationResults object with all metrics
    """
    evaluator = ModelEvaluator(tokenizer)
    return evaluator.evaluate_comprehensive(model, data, language, num_samples) 