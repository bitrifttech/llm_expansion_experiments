# Continual Learning Utilities

This package provides shared utilities for analyzing models and calculating metrics across different continual learning experiments.

## CodeSearchNetDataLoader

The `CodeSearchNetDataLoader` class provides consistent data loading and preparation for all continual learning experiments, ensuring fair comparison between different approaches.

### Features

- **🔄 Consistent Splits**: Ensures all experiments use identical data splits
- **📊 Multiple Formats**: Supports HuggingFace datasets, dict format, and raw format
- **🎯 Reproducible**: Uses fixed seeds for reproducible data loading
- **✅ Validation**: Built-in data consistency validation
- **📈 Statistics**: Provides data statistics and summaries
- **🔧 Flexible**: Configurable train/validation split sizes

### Usage

#### Drop-in Replacement (Recommended)
```python
from utils.data_loader import load_and_prepare_data

# Exact same interface as original experiments
python_train, python_val, js_train, js_val = load_and_prepare_data()

# With custom sizes
python_train, python_val, js_train, js_val = load_and_prepare_data(
    python_train_size=15000,
    python_val_size=5000,
    js_train_size=15000,
    js_val_size=5000,
    format_type="huggingface"
)
```

#### Class-based Usage (More Control)
```python
from utils.data_loader import CodeSearchNetDataLoader

# Create loader with custom configuration
loader = CodeSearchNetDataLoader(
    python_train_size=15000,
    python_val_size=5000,
    js_train_size=15000,
    js_val_size=5000,
    seed=42
)

# Load data in different formats
python_train, python_val, js_train, js_val = loader.load_data(format_type="huggingface")

# Get statistics
stats = loader.get_data_stats()
print(f"Total samples: {stats['total_samples']}")

# Validate consistency
is_consistent = loader.validate_data_consistency()
```

### Supported Formats

#### 1. HuggingFace Format (`format_type="huggingface"`)
Returns HuggingFace Dataset objects with original CodeSearchNet fields:
```python
# Returns: datasets.Dataset objects
python_train, python_val, js_train, js_val = loader.load_data("huggingface")
sample = python_train[0]  # Access: sample['func_code_string'], sample['func_documentation_string']
```

#### 2. Dict Format (`format_type="dict"`)
Returns lists of dictionaries with `input`/`target` keys for training:
```python
# Returns: List[Dict] with 'input' and 'target' keys
python_train, python_val, js_train, js_val = loader.load_data("dict")
sample = python_train[0]  # Access: sample['input'], sample['target']
```

#### 3. Raw Format (`format_type="raw"`)
Returns lists of dictionaries with original CodeSearchNet fields:
```python
# Returns: List[Dict] with original fields
python_train, python_val, js_train, js_val = loader.load_data("raw")
sample = python_train[0]  # Access: sample['func_name'], sample['docstring'], sample['code']
```

### Integration in Experiments

Replace existing `load_and_prepare_data()` functions:

```python
# OLD WAY (inconsistent across experiments)
def load_and_prepare_data():
    dataset = load_dataset("code_search_net", split="train")
    python_data = dataset.filter(lambda x: x["language"] == "python").select(range(20000))
    # ... different implementations in each experiment

# NEW WAY (consistent across all experiments)
from utils.data_loader import load_and_prepare_data

python_train, python_val, js_train, js_val = load_and_prepare_data(
    python_train_size=15000,
    python_val_size=5000,
    js_train_size=15000,
    js_val_size=5000,
    format_type="huggingface"  # or "dict" or "raw"
)
```

### Benefits

- **🔄 Consistency**: All experiments use identical data splits
- **🎯 Fair Comparison**: Eliminates data-related variables between approaches
- **⚡ Efficiency**: Cached loading and format conversion
- **🔧 Flexibility**: Multiple formats for different experiment needs
- **✅ Reliability**: Built-in validation and error handling
- **📊 Transparency**: Clear statistics and data summaries

## ModelAnalyzer

The `ModelAnalyzer` class provides comprehensive analysis of transformer models including:

### Features

- **📊 Model Overview**: Basic model information, architecture type, device, and data type
- **🔢 Parameter Analysis**: Total, trainable, and frozen parameter counts with percentages
- **🏗️ Architecture Details**: Layer counts, dimensions, and model-specific information
- **🔧 Custom Component Detection**: Automatically detects LoRA, ExpandedFFN, and other custom layers
- **⚡ Efficiency Metrics**: Parameters per MB, memory per parameter, trainable ratios
- **🎯 Layer Breakdown**: Detailed analysis of individual layers with parameter counts
- **🔄 Model Comparison**: Before/after analysis for model modifications

### Usage

#### Basic Analysis
```python
from utils.model_analyzer import ModelAnalyzer, analyze_model

# Simple analysis
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")
analysis = analyze_model(model, "CodeT5-Small", detailed=True)

# Using the class directly
analyzer = ModelAnalyzer(model, "My Model")
analysis = analyzer.analyze(detailed=True)
```

#### Model Comparison
```python
# Compare original vs modified model
original_analyzer = ModelAnalyzer(original_model, "Original")
modified_analyzer = ModelAnalyzer(modified_model, "Modified")

comparison = original_analyzer.compare_with(modified_analyzer, "My Modification")
```

### Output Example

```
============================================================
🔍 ANALYZING MODEL: CodeT5-Small (Base)
============================================================

📊 MODEL OVERVIEW
├── Model: CodeT5-Small (Base)
├── Type: T5ForConditionalGeneration
├── Architecture: Encoder-Decoder
├── Device: mps:0
└── Data Type: torch.float32

🔢 PARAMETER SUMMARY
├── Total Parameters: 60,492,288
├── Trainable Parameters: 60,492,288 (100.00%)
├── Frozen Parameters: 0 (0.00%)
└── Memory Usage: 461.52 MB

🏗️ ARCHITECTURE DETAILS
├── Encoder Layers: 6
├── Decoder Layers: 6
├── Model Dimension: 512
└── FFN Dimension: 2048

⚡ EFFICIENCY METRICS
├── Parameters per MB: 131,072
├── Memory per Parameter: 8.00 bytes
└── Trainable Ratio: 100.00%

🎯 TOP TRAINABLE LAYERS
├── shared: 16,435,200 params (Embedding)
├── lm_head: 16,435,200 params (Linear)
├── encoder.block.0.layer.1.DenseReluDense.wi: 1,048,576 params (Linear)
├── encoder.block.0.layer.1.DenseReluDense.wo: 1,048,576 params (Linear)
└── encoder.block.1.layer.1.DenseReluDense.wi: 1,048,576 params (Linear)
    ... and 127 more trainable layers
============================================================
```

### Custom Component Detection

The analyzer automatically detects and highlights custom components:

- **ExpandedFFN**: FFN expansion layers
- **LoRA**: Low-rank adaptation layers
- **PeftModel**: Parameter-efficient fine-tuning models
- **AdaLoRA**: Adaptive LoRA layers
- **IA3**: Infused Adapter layers

### Integration in Experiments

The ModelAnalyzer is integrated into all continual learning experiments:

1. **Base Model Analysis**: Analyze the original model before modifications
2. **Expansion Analysis**: Show detailed comparison after adding parameters
3. **Training Analysis**: Track parameter changes during training
4. **Cross-Experiment Consistency**: Ensure consistent metrics across all approaches

### Benefits

- **🔄 Consistency**: Standardized analysis across all experiments
- **📈 Insights**: Detailed parameter and memory breakdowns
- **🎯 Focus**: Highlights trainable components and custom layers
- **⚡ Efficiency**: Quick comparison of different approaches
- **🔧 Debugging**: Easy identification of model modifications

## Future Extensions

- **Memory Profiling**: Runtime memory usage tracking
- **FLOP Counting**: Computational complexity analysis
- **Gradient Analysis**: Gradient flow and magnitude tracking
- **Performance Metrics**: Training speed and convergence analysis 