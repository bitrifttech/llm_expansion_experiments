# Hybrid LoRA + Full Layer Continual Learning Experiment

This experiment explores the interaction between LoRA adapters and full transformer layers in continual learning scenarios, testing both task-specific and shared component strategies.

## Experiment Design

### **Experiment 1: Task-Specific Components**
```
Training Sequence:
1. Base Model (frozen) + LoRA Adapter (Python) + Full Layer (Python)
2. Base Model (frozen) + LoRA Adapter (JavaScript) + Full Layer (JavaScript)

Architecture: Base + Task-specific LoRA + Task-specific Full Layer
```

### **Experiment 2: Shared Full Layer**
```
Training Sequence:
1. Base Model (frozen) + LoRA Adapter (Python) + Full Layer (Python)
2. Base Model (frozen) + LoRA Adapter (JavaScript) + Full Layer (Python, reused)

Architecture: Base + Task-specific LoRA + Shared Full Layer
```

## Research Questions

1. **Component Interaction**: How do LoRA adapters and full layers interact when combined?
2. **Shared vs. Specific**: Does sharing full layers across tasks help or hurt performance?
3. **Capacity Allocation**: How does the combination affect parameter efficiency?
4. **Continual Learning**: Which approach better preserves knowledge and enables transfer?

## Expected Outcomes

### **Hypotheses**:
- **Experiment 1**: Task-specific components may provide better individual task performance
- **Experiment 2**: Shared full layer might enable better cross-task knowledge transfer
- **LoRA + Full Layer**: Combined approach might outperform either method alone

### **Metrics to Compare**:
- Task-specific performance (BLEU, METEOR, AST similarity)
- Cross-task transfer and interference
- Parameter efficiency vs. performance trade-offs
- Training time and memory usage

## Files Structure

- `hybrid_experiment.py` - Main experiment script
- `test_setup.py` - Setup verification and testing script
- `experiment_1_results/` - Task-specific components results
- `experiment_2_results/` - Shared full layer results
- `analysis.md` - Detailed results analysis and comparison (generated after run)

## Usage

### 1. Test Setup
First, verify that everything is configured correctly:
```bash
cd hybrid_lora_full_layer_experiment
python test_setup.py
```

### 2. Run Experiment
If tests pass, run the full experiment:
```bash
python hybrid_experiment.py
```

### 3. Monitor Progress
The experiment will log detailed progress including:
- Model architecture creation
- Training loss progression
- Evaluation metrics
- Memory and time usage

### 4. Results
Results will be saved to:
- `hybrid_experiment_results.json` - Comprehensive results in JSON format
- `experiment_1_results/` - Task-specific model artifacts
- `experiment_2_results/` - Shared layer model artifacts

## Experimental Setup

- **Base Model**: Salesforce/codet5-small
- **Dataset**: CodeSearchNet (Python & JavaScript, 8K train + 2K val each)
- **Training**: Sequential learning (Python → JavaScript)
- **LoRA Config**: Rank=16, Alpha=32, Dropout=0.1
- **Full Layer**: Additional transformer layer (4.94% of parameters)
- **Training**: 2 epochs, batch size 8, learning rate 5e-4

## Key Differences from Previous Experiment

1. **Hybrid Architecture**: Combines LoRA + Full Layer (not either/or)
2. **Shared Component Testing**: Tests shared vs. task-specific full layers
3. **Component Interaction Analysis**: Studies how different parameter-efficient methods interact
4. **Capacity Analysis**: Examines optimal allocation between LoRA and full layers

## Hardware Requirements

- **Recommended**: CUDA GPU with 16GB+ VRAM
- **Minimum**: 8GB GPU memory or Apple Silicon MPS
- **Fallback**: CPU (slower training)
- **System Memory**: 16GB+ RAM recommended

## Expected Runtime

- **GPU (CUDA)**: ~30-45 minutes total
- **Apple Silicon (MPS)**: ~60-90 minutes total  
- **CPU**: 3-5 hours total

## Troubleshooting

### Common Issues:
1. **CUDA out of memory**: Reduce batch size or use smaller model
2. **Import errors**: Ensure all dependencies are installed
3. **Dataset loading fails**: Check internet connection
4. **Slow training**: Verify GPU/MPS is being used

### Debug Mode:
Add debug logging by modifying the log level in the script:
```python
log_message("Debug info", level="DEBUG")
```

## Results Interpretation

### Key Metrics:
- **BLEU Score**: Code generation quality (0-1, higher better)
- **Pass Rate**: Syntactic correctness (0-1, higher better)  
- **Forgetting Rate**: Knowledge retention (lower better)
- **Training Time**: Efficiency measure
- **Memory Usage**: Resource consumption

### Expected Insights:
- Performance comparison between shared vs. task-specific architectures
- Parameter efficiency analysis of hybrid approaches
- Continual learning effectiveness measurement
- Optimal component allocation strategies 