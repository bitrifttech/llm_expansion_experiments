# Continual Learning Utilities

This package provides shared utilities for analyzing models and calculating metrics across different continual learning experiments.

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