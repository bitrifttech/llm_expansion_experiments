# Layer Widening Continual Learning Experiment

This experiment explores **FFN (Feed-Forward Network) expansion** as a novel parameter-efficient continual learning approach. Instead of adding new layers (Full Layer) or low-rank adaptations (LoRA), this method expands the width of existing transformer layers by adding trainable parameters while keeping original parameters frozen.

## Approach: FFN Expansion

### **Core Concept**
```
Original FFN:     Input → [512 → 2048 → 512] → Output
Expanded FFN:     Input → [512 → 2048 → 512] (frozen) + [512 → 512 → 512] (trainable) → Output
                         ↑                                ↑
                    Original Path                   Expansion Path
                     (frozen)                       (trainable)
```

### **Key Innovation**
- **Layer Widening**: Expand existing layers rather than adding new ones
- **Parallel Expansion**: Add trainable paths parallel to frozen original paths
- **Residual Integration**: Combine original and expansion outputs with residual connections
- **Parameter Efficiency**: Target ~1.5M parameters (2.4% of model)

## Research Questions

1. **Efficiency**: Is FFN expansion more parameter-efficient than LoRA/Full Layer approaches?
2. **Performance**: Can it match or exceed existing methods' performance?
3. **Continual Learning**: How does it handle catastrophic forgetting?
4. **Scalability**: How does performance scale with expansion size?

## Expected Outcomes

### **Hypotheses**
- FFN expansion will provide better parameter efficiency than Full Layer addition
- Performance will be competitive with LoRA while using similar parameter count
- Catastrophic forgetting will be moderate (3-5%)
- Training will be efficient due to focused parameter updates

### **Target Metrics**
- **Performance**: BLEU score ≥ 0.235 (competitive with LoRA)
- **Efficiency**: ~1.5M parameters (2.4% of model)
- **Forgetting**: Catastrophic forgetting < 5%
- **Training**: Total time < 25 minutes

## Files Structure

- `ffn_expansion_continual_learning.py` - Main experiment implementation
- `test_setup.py` - Setup verification and testing script
- `README.md` - This documentation
- `ffn_expansion_python/` - Python task model artifacts (generated)
- `ffn_expansion_javascript/` - JavaScript task model artifacts (generated)
- `ffn_expansion_experiment_results.json` - Comprehensive results (generated)

## Technical Implementation

### **ExpandedFFN Module**
```python
class ExpandedFFN(torch.nn.Module):
    def __init__(self, original_ffn, expansion_size=512):
        # Freeze original FFN
        self.original_ffn = original_ffn  # Frozen
        
        # Add trainable expansion path
        self.expansion_up = nn.Linear(input_dim, expansion_size)
        self.expansion_down = nn.Linear(expansion_size, output_dim)
        
    def forward(self, x):
        original_out = self.original_ffn(x)  # Frozen path
        expanded_out = self.expansion_down(
            self.activation(self.expansion_up(x))
        )  # Trainable path
        return original_out + expanded_out  # Residual combination
```

### **Model Expansion Process**
1. **Load Base Model**: Salesforce/codet5-small
2. **Freeze All Parameters**: Preserve original knowledge
3. **Replace FFN Layers**: Substitute with ExpandedFFN modules
4. **Enable Expansion Training**: Only expansion parameters are trainable
5. **Sequential Training**: Python → JavaScript (fresh model each time)

## Usage

### 1. Test Setup
First, verify that everything works correctly:
```bash
cd layer_widening_continual_learning_experiment
python test_setup.py
```

### 2. Run Experiment
If tests pass, run the full experiment:
```bash
python ffn_expansion_continual_learning.py
```

### 3. Monitor Progress
The experiment will log detailed progress including:
- FFN expansion process
- Training loss progression
- Evaluation metrics
- Memory and time usage
- Parameter counts

### 4. Results
Results will be saved to:
- `ffn_expansion_experiment_results.json` - Comprehensive metrics
- `ffn_expansion_python/` - Python task model
- `ffn_expansion_javascript/` - JavaScript task model

## Experimental Setup

- **Base Model**: Salesforce/codet5-small (63.8M parameters)
- **Dataset**: CodeSearchNet (Python & JavaScript, 8K train + 2K val each)
- **Training**: Sequential learning (Python → JavaScript, fresh models)
- **Expansion Size**: 512 (25% increase in FFN intermediate size)
- **Training**: 2 epochs, batch size 8, learning rate 5e-4
- **Evaluation**: BLEU, Pass Rate, METEOR, Edit Distance, AST Similarity

## Key Differences from Previous Experiments

1. **Layer Widening**: Expands existing layers instead of adding new ones
2. **Parallel Paths**: Trainable paths run parallel to frozen original paths
3. **Focused Expansion**: Targets FFN layers specifically (most parameters)
4. **Fresh Models**: Each task gets a fresh expanded model (no interference)
5. **Residual Integration**: Combines original and expansion outputs

## Parameter Efficiency Analysis

| Component | Original Size | Expansion | New Parameters | Percentage |
|-----------|---------------|-----------|----------------|------------|
| **Encoder FFN** | 6 layers × 2048 | +512 each | ~786K | 1.2% |
| **Decoder FFN** | 6 layers × 2048 | +512 each | ~786K | 1.2% |
| **Total Expansion** | - | - | **~1.57M** | **2.4%** |

### **Comparison with Other Approaches**
- **LoRA**: 1.78M parameters (2.8%) - Similar efficiency
- **Full Layer**: 3.15M parameters (4.9%) - Less efficient
- **Hybrid**: 4.92M parameters (7.7%) - Much less efficient

## Hardware Requirements

- **Recommended**: CUDA GPU with 8GB+ VRAM
- **Minimum**: 4GB GPU memory or Apple Silicon MPS
- **Fallback**: CPU (slower training)
- **System Memory**: 8GB+ RAM recommended

## Expected Runtime

- **GPU (CUDA)**: ~20-25 minutes total
- **Apple Silicon (MPS)**: ~30-40 minutes total
- **CPU**: 2-3 hours total

## Troubleshooting

### Common Issues:
1. **Import errors**: Run `test_setup.py` first to verify dependencies
2. **CUDA out of memory**: Reduce batch size or expansion size
3. **Model loading fails**: Check internet connection for model download
4. **Slow training**: Verify GPU/MPS is being used

### Debug Mode:
Enable detailed logging by modifying the expansion size:
```python
# In ffn_expansion_continual_learning.py
expansion_size = 256  # Reduce for testing
```

## Results Interpretation

### Key Metrics:
- **BLEU Score**: Code generation quality (0-1, higher better)
- **Pass Rate**: Syntactic correctness (0-1, higher better)
- **Forgetting Rate**: Knowledge retention (lower better)
- **Expansion Parameters**: Number of trainable parameters added
- **Training Time**: Efficiency measure

### Expected Insights:
- Parameter efficiency comparison with LoRA/Full Layer approaches
- Catastrophic forgetting analysis for layer widening
- Training efficiency of focused parameter updates
- Scalability of expansion size vs. performance

## Comparison with Previous Experiments

This experiment will be compared against:

| Approach | Parameters | Avg BLEU | Forgetting | Training Time |
|----------|------------|----------|------------|---------------|
| **LoRA** | 1.78M (2.8%) | 0.232 | 0.78% | ~25 min |
| **Full Layer** | 3.15M (4.9%) | 0.210 | -6.20% | ~30 min |
| **Hybrid** | 4.92M (7.7%) | 0.250 | 4.38% | ~16 min |
| **FFN Expansion** | ~1.5M (2.4%) | **TBD** | **TBD** | **TBD** |

### Success Criteria:
- **Minimum**: Performance within 5% of LoRA, forgetting < 10%
- **Target**: Performance equal to LoRA, parameters < 2M
- **Exceptional**: Performance approaching Hybrid, best efficiency

## Future Extensions

1. **Different Expansion Strategies**: Attention heads, embeddings
2. **Adaptive Expansion**: Dynamic expansion based on task complexity
3. **Hybrid Combinations**: FFN Expansion + LoRA
4. **Multi-Task**: More programming languages
5. **Theoretical Analysis**: Mathematical framework for layer widening

---

**Experiment Status**: ✅ Ready to run  
**Expected Impact**: Novel parameter-efficient continual learning method with potential for superior efficiency-performance trade-offs 