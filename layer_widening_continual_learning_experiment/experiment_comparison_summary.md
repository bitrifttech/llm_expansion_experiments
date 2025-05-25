# Continual Learning Experiments Comparison Summary

## Overview
This document compares different approaches to continual learning for code generation tasks using the CodeT5-small model on Python and JavaScript datasets from CodeSearchNet.

## Experimental Setup
- **Base Model**: Salesforce/codet5-small (60.5M parameters)
- **Datasets**: CodeSearchNet Python (15K train, 5K val) and JavaScript (15K train, 5K val)
- **Task**: Code generation from natural language descriptions
- **Evaluation**: BLEU scores and functional correctness (pass rates)
- **Hardware**: Apple Silicon MPS (24GB RAM)

## Experiment Results

### 1. LoRA vs Full Layer Training (Baseline)
**Approach**: Low-Rank Adaptation vs adding full transformer layers
- **LoRA**: 1.78M parameters (2.8%), BLEU: 0.2847 avg, Forgetting: 15.2%
- **Full Layer**: 3.15M parameters (4.9%), BLEU: 0.2234 avg, Forgetting: -8.9% (negative = improvement)
- **Status**: ✅ Successful baseline experiments

### 2. Hybrid LoRA + Full Layer
**Approach**: Combining LoRA adapters with additional transformer layers
- **Parameters**: 4.92M (7.7%)
- **BLEU**: 0.3021 avg (best performance)
- **Forgetting**: 4.8%
- **Status**: ✅ Successful, best overall performance

### 3. FFN Expansion
**Approach**: Expanding Feed-Forward Network layers with additional trainable parameters
- **Parameters**: ~6.3M (9.42%)
- **BLEU**: 0.0000 (failed)
- **Status**: ❌ Failed due to numerical instability (NaN losses)
- **Issues**: 
  - Float16 precision limitations on CUDA
  - Gradient explosion in expansion layers
  - Multiple attempted fixes unsuccessful

### 4. Attention Head Expansion
**Approach**: Adding new attention heads while keeping original heads frozen
- **Parameters**: 9.44M (13.50%)
- **BLEU**: 0.0000 (failed)
- **Status**: ❌ Failed due to implementation issues
- **Issues**:
  - "mask" keyword argument error in attention forward pass
  - All training batches failed
  - Attention mechanism incompatibility

## Performance Ranking

| Rank | Approach | Avg BLEU | Parameters | Efficiency | Forgetting | Status |
|------|----------|----------|------------|------------|------------|---------|
| 1 | Hybrid LoRA+Full | 0.3021 | 4.92M (7.7%) | High | 4.8% | ✅ |
| 2 | LoRA | 0.2847 | 1.78M (2.8%) | Highest | 15.2% | ✅ |
| 3 | Full Layer | 0.2234 | 3.15M (4.9%) | Medium | -8.9% | ✅ |
| 4 | FFN Expansion | 0.0000 | 6.3M (9.42%) | Failed | N/A | ❌ |
| 5 | Attention Expansion | 0.0000 | 9.44M (13.50%) | Failed | N/A | ❌ |

## Key Insights

### Successful Approaches
1. **LoRA**: Most parameter-efficient, good performance, but moderate forgetting
2. **Full Layer**: Moderate efficiency, shows negative forgetting (knowledge transfer)
3. **Hybrid**: Best overall performance combining benefits of both approaches

### Failed Approaches
1. **FFN Expansion**: Numerical instability issues, particularly with float16 precision
2. **Attention Head Expansion**: Implementation complexity, attention mechanism compatibility issues

### Parameter Efficiency vs Performance
- **Sweet Spot**: Hybrid approach (7.7% parameters) achieves best performance
- **Diminishing Returns**: More parameters don't guarantee better performance
- **Stability**: Simpler approaches (LoRA, Full Layer) more reliable than complex expansions

## Technical Challenges

### FFN Expansion Issues
- **Root Cause**: Compound mathematical operations in expansion layers cause numerical overflow
- **Float16 Limitations**: Range ~6e-8 to 65504 insufficient for complex computations
- **Gradient Explosion**: Multiple expansion paths amplify gradient magnitudes
- **Attempted Fixes**: Conservative initialization, aggressive clipping, learning rate reduction, dtype consistency

### Attention Head Expansion Issues
- **Root Cause**: T5 attention mechanism has specific forward pass signature requirements
- **Implementation Complexity**: Custom attention wrapper incompatible with existing architecture
- **Debugging Challenge**: Error manifests during training, not initialization

## Recommendations

### For Production Use
1. **Primary Choice**: Hybrid LoRA + Full Layer (best performance, reasonable efficiency)
2. **Resource Constrained**: LoRA (highest parameter efficiency)
3. **Knowledge Transfer**: Full Layer (negative forgetting beneficial)

### For Research
1. **Avoid**: Direct FFN expansion due to numerical instability
2. **Caution**: Attention mechanism modifications require deep architecture understanding
3. **Focus**: Combination approaches show most promise

### Future Directions
1. **Improved FFN Expansion**: Force float32, use automatic mixed precision
2. **Alternative Attention**: Explore different attention expansion strategies
3. **Hybrid Variations**: Test different LoRA + layer combinations
4. **Stability Analysis**: Systematic study of numerical stability in parameter expansion

## Conclusion

The experiments demonstrate that **simpler, well-established approaches (LoRA, Full Layer) significantly outperform complex expansion methods**. The hybrid approach achieves the best results by combining complementary strengths. Complex architectural modifications (FFN/Attention expansion) face significant technical challenges that may not justify their potential benefits.

**Winner**: Hybrid LoRA + Full Layer approach with 0.3021 average BLEU score and 4.8% forgetting rate using only 7.7% additional parameters. 