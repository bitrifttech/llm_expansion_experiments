# Continual Learning: LoRA vs Full Layer Training Experiment

This directory contains a comprehensive experiment comparing two approaches for continual learning in code generation models:

## Approaches Compared

1. **LoRA (Low-Rank Adaptation)**: 
   - Adds small trainable adapter matrices (~0.1% of parameters)
   - Frozen base model + swappable adapter matrices

2. **Full Layer Training**: 
   - Adds entire new transformer layers (~5% of parameters)  
   - Frozen base model + swappable transformer layers

## Files and Directories

- `lora_vs_full_layer_training_mac_and_cuda.py` - Main experiment script
- `results_analysis.md` - **📊 Comprehensive analysis of experimental results and findings**
- `lora_python/` - LoRA adapter for Python code generation
- `lora_javascript/` - LoRA adapter for JavaScript code generation  
- `full_python/` - Full layer checkpoint for Python code generation
- `experimental_results.json` - Detailed experimental results (if generated)

## Experiment Details

- **Base Model**: Salesforce/codet5-small
- **Dataset**: CodeSearchNet (Python & JavaScript)
- **Training**: Sequential learning (Python → JavaScript)
- **Evaluation**: BLEU, METEOR, AST similarity, pass rates
- **Focus**: Measuring catastrophic forgetting and continual learning effectiveness

## Key Findings

Both approaches successfully avoid catastrophic forgetting with task-specific component swapping:
- **LoRA**: Better task-specific performance (15.9% superior JavaScript BLEU), minimal forgetting (0.78%)
- **Full Layer**: Better cross-task knowledge retention, **negative forgetting (-6.2% = improvement!)**

## Quick Results

| Approach | Python BLEU | JavaScript BLEU | Training Time | Memory | Forgetting |
|----------|-------------|-----------------|---------------|--------|------------|
| LoRA | 0.2150 | **0.2489** | 22.97 min | 1.77 GB | 0.78% |
| Full Layer | 0.2046 | 0.2147 | **21.69 min** | **0.01 GB** | **-6.20%** |

📖 **See `results_analysis.md` for comprehensive analysis, detailed findings, and theoretical implications.**

## Usage

Run the experiment:
```bash
python lora_vs_full_layer_training_mac_and_cuda.py
```

Requires CUDA/MPS GPU and ~16GB VRAM for optimal performance.

## Results Summary

From the latest experimental run:

### LoRA Results
- Final Python BLEU: 0.2150
- Final JavaScript BLEU: 0.2489  
- Training Time: 22.97 minutes
- Memory Usage: 1.77 GB
- Python Forgetting: 0.78%

### Full Layer Results
- Final Python BLEU: 0.2046
- Final JavaScript BLEU: 0.2147
- Training Time: 21.69 minutes  
- Memory Usage: 0.01 GB
- Python Forgetting: -6.20% (improvement) 