# Model Evaluator Utility

A comprehensive, reusable evaluation system for code generation models that consolidates evaluation logic from various continual learning experiments.

## Overview

The `ModelEvaluator` utility provides consistent, comprehensive evaluation capabilities across all experiments. It replaces the scattered evaluation code found in individual experiments with a unified, well-tested, and feature-rich evaluation system.

## Key Features

### 🎯 **Comprehensive Metrics**
- **BLEU Score**: Standard text similarity metric
- **METEOR Score**: Advanced semantic similarity metric
- **Pass Rate**: Syntactic correctness (compilation/parsing)
- **AST Similarity**: Structural code similarity
- **Edit Distance**: Character-level similarity
- **Code Complexity**: Cyclomatic complexity analysis
- **Composite Score**: Weighted combination of all metrics

### 🔧 **Configurable Evaluation**
- Customizable generation parameters (beam size, temperature, etc.)
- Adjustable metric weights for composite scoring
- Language-specific evaluation strategies
- Flexible data format handling

### 🧠 **Continual Learning Support**
- Forward transfer measurement
- Backward interference detection
- Retention score calculation
- Forgetting rate analysis
- Complete continual learning experiment evaluation

### 🛡️ **Robust Error Handling**
- Graceful handling of invalid code
- Protection against edge cases
- Comprehensive logging and debugging support
- Zero-division protection

### 🔄 **Backward Compatibility**
- Drop-in replacement for existing evaluation code
- Convenience functions for simple migration
- Consistent API across all experiments

## Installation

The evaluator is part of the `utils` package and requires the following dependencies:

```python
# Core dependencies (already in requirements.txt)
torch
transformers
nltk
numpy
datasets
```

## Quick Start

### Basic Usage

```python
from utils.model_evaluator import ModelEvaluator
from transformers import AutoTokenizer

# Initialize
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
evaluator = ModelEvaluator(tokenizer)

# Basic evaluation (backward compatible)
bleu, pass_rate = evaluator.evaluate_basic(model, data, "python", 100)
print(f"BLEU: {bleu:.4f}, Pass Rate: {pass_rate:.2%}")

# Comprehensive evaluation
results = evaluator.evaluate_comprehensive(model, data, "python", 100)
print(f"BLEU: {results.bleu:.4f}")
print(f"METEOR: {results.meteor:.4f}")
print(f"AST Similarity: {results.ast_similarity:.4f}")
print(f"Composite Score: {results.composite_score(evaluator.config):.4f}")
```

### Advanced Configuration

```python
from utils.model_evaluator import ModelEvaluator, EvaluationConfig

# Custom configuration
config = EvaluationConfig(
    max_length=1024,           # Longer generation
    num_beams=5,               # Higher quality
    temperature=0.5,           # Less randomness
    default_num_samples=200,   # More samples
    bleu_weight=0.4,           # Emphasize BLEU
    ast_similarity_weight=0.3  # Emphasize structure
)

evaluator = ModelEvaluator(tokenizer, config)
results = evaluator.evaluate_comprehensive(model, data, "python")
```

### Continual Learning Evaluation

```python
from utils.model_evaluator import ModelEvaluator, ContinualLearningEvaluator

# Initialize evaluators
base_evaluator = ModelEvaluator(tokenizer)
cl_evaluator = ContinualLearningEvaluator(base_evaluator)

# Complete continual learning evaluation
results = cl_evaluator.evaluate_continual_learning_experiment(
    model_before_training=base_model,
    model_after_task1=model_after_python,
    model_after_task2=model_after_js,
    task1_data=python_data,
    task2_data=js_data,
    task1_language="python",
    task2_language="javascript",
    num_samples=100
)

# Access continual learning metrics
cl_metrics = results['continual_learning_metrics']
print(f"Forward Transfer: {cl_metrics['forward_transfer']:+.4f}")
print(f"Backward Interference: {cl_metrics['backward_interference']:+.4f}")
print(f"Forgetting Rate: {cl_metrics['forgetting_rate']:.2%}")
```

## Migration Guide

### From Existing Experiments

**Old Code (from experiments):**
```python
def evaluate_model(self, model, data, num_samples: int = 100, language: str = None):
    model.eval()
    bleu_scores = []
    pass_scores = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(data))):
            # ... 50+ lines of complex evaluation logic ...
            bleu_scores.append(bleu_score)
            pass_scores.append(pass_score)
    
    return {
        'bleu': np.mean(bleu_scores),
        'pass_rate': np.mean(pass_scores)
    }
```

**New Code (using ModelEvaluator):**
```python
from utils.model_evaluator import ModelEvaluator

def evaluate_model(self, model, data, num_samples: int = 100, language: str = None):
    evaluator = ModelEvaluator(self.tokenizer)
    results = evaluator.evaluate_comprehensive(model, data, language, num_samples)
    return results.to_dict()  # Returns ALL metrics, not just BLEU and pass rate
```

### Migration Benefits

✅ **Consistent evaluation** across all experiments  
✅ **More metrics** (METEOR, AST similarity, complexity, etc.)  
✅ **Better error handling** and edge case management  
✅ **Configurable generation** parameters  
✅ **Language auto-detection**  
✅ **Continual learning metrics** built-in  
✅ **Backward compatibility** with existing code  
✅ **Comprehensive testing** and validation  

## API Reference

### Classes

#### `EvaluationConfig`
Configuration dataclass for evaluation parameters.

**Parameters:**
- `max_length: int = 512` - Maximum generation length
- `num_beams: int = 3` - Number of beams for generation
- `temperature: float = 0.7` - Generation temperature
- `default_num_samples: int = 100` - Default number of samples to evaluate
- `input_fraction: float = 0.5` - Fraction of source code to use as input
- Metric weights for composite scoring
- Language detection keywords

#### `EvaluationResults`
Results dataclass containing all evaluation metrics.

**Attributes:**
- `bleu: float` - BLEU score (0-1)
- `meteor: float` - METEOR score (0-1)
- `pass_rate: float` - Syntactic correctness rate (0-1)
- `edit_distance: float` - Normalized edit distance (0-1)
- `ast_similarity: float` - AST structural similarity (0-1)
- `complexity: float` - Code complexity score
- `num_samples: int` - Number of samples evaluated
- `language: str` - Detected/specified language

**Methods:**
- `composite_score(config)` - Calculate weighted composite score
- `to_dict()` - Convert to dictionary for serialization

#### `ModelEvaluator`
Main evaluation class with comprehensive metrics.

**Methods:**
- `evaluate_comprehensive(model, data, language=None, num_samples=None)` - Full evaluation
- `evaluate_basic(model, data, language=None, num_samples=None)` - Basic BLEU + pass rate

#### `ContinualLearningEvaluator`
Specialized evaluator for continual learning experiments.

**Methods:**
- `calculate_continual_learning_metrics(...)` - Calculate CL-specific metrics
- `evaluate_continual_learning_experiment(...)` - Complete CL evaluation

### Convenience Functions

```python
# Backward compatible functions
evaluate_model_basic(model, data, tokenizer, language=None, num_samples=100)
evaluate_model_comprehensive(model, data, tokenizer, language=None, num_samples=100)
```

## Supported Languages

- **Python**: Full AST analysis, compilation checking
- **JavaScript**: Token-based analysis, basic syntax checking
- **Auto-detection**: Automatic language detection from code content
- **Generic**: Fallback evaluation for unknown languages

## Configuration Examples

### Fast Evaluation (for development/testing)
```python
fast_config = EvaluationConfig(
    max_length=128,
    num_beams=1,
    do_sample=False,
    default_num_samples=50
)
```

### High-Quality Evaluation (for final results)
```python
quality_config = EvaluationConfig(
    max_length=1024,
    num_beams=5,
    temperature=0.5,
    default_num_samples=500,
    bleu_weight=0.4,
    ast_similarity_weight=0.3
)
```

### Continual Learning Focus
```python
cl_config = EvaluationConfig(
    default_num_samples=200,
    bleu_weight=0.25,
    meteor_weight=0.25,
    ast_similarity_weight=0.25,
    pass_rate_weight=0.25
)
```

## Data Format Support

The evaluator automatically handles multiple data formats:

```python
# Dictionary format (most common)
data = [{"func_code_string": "def test(): pass"}, ...]

# Alternative dictionary fields
data = [{"code": "def test(): pass"}, ...]
data = [{"source_code": "def test(): pass"}, ...]

# String format
data = ["def test(): pass", ...]

# Object format (with attributes)
data = [obj_with_func_code_string_attr, ...]
```

## Testing

Run the comprehensive test suite:

```bash
cd utils
python test_model_evaluator.py
```

The test suite covers:
- All evaluation metrics
- Edge cases and error handling
- Different data formats
- Language detection
- Configuration options
- Continual learning metrics

## Performance Considerations

### Memory Usage
- Evaluation is done in batches with `torch.no_grad()`
- Models are automatically set to eval mode
- GPU memory is efficiently managed

### Speed Optimization
- Use smaller `num_beams` for faster evaluation
- Reduce `max_length` for shorter generation
- Set `do_sample=False` for deterministic, faster generation
- Use smaller `num_samples` for quick testing

### Quality vs Speed Trade-offs
```python
# Fast but lower quality
fast_config = EvaluationConfig(num_beams=1, max_length=256)

# Slow but higher quality  
quality_config = EvaluationConfig(num_beams=5, max_length=1024)
```

## Troubleshooting

### Common Issues

**1. NLTK Data Missing**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

**2. Memory Issues**
- Reduce `num_samples` or `max_length`
- Use smaller beam size
- Ensure `torch.no_grad()` context

**3. Language Detection Issues**
- Explicitly specify language parameter
- Check that code contains recognizable keywords
- Use "unknown" for generic evaluation

**4. Zero Scores**
- Check data format and field names
- Verify model generates non-empty output
- Check for syntax errors in generated code

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# The evaluator will print detailed information about each evaluation step
```

## Contributing

When adding new metrics or features:

1. Add the metric calculation method to `ModelEvaluator`
2. Update `EvaluationResults` dataclass
3. Add comprehensive tests in `test_model_evaluator.py`
4. Update this documentation
5. Add examples in `demo_model_evaluator.py`

## Examples

See `demo_model_evaluator.py` for comprehensive usage examples including:
- Basic evaluation
- Comprehensive evaluation with all metrics
- Continual learning evaluation
- Configuration options
- Migration examples

Run the demo:
```bash
cd utils
python demo_model_evaluator.py
```

## Integration with Experiments

The evaluator is designed to be a drop-in replacement for existing evaluation code in:

- `hybrid_lora_full_layer_experiment/`
- `lora_vs_full_layer_training/`
- `layer_widening_continual_learning_experiment/`

Simply replace the existing `evaluate_model` methods with calls to `ModelEvaluator` for consistent, comprehensive evaluation across all experiments. 