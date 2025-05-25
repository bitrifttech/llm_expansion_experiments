# FFN Expansion Experiment - Fixes for Fair Comparison

## Issues Identified and Fixed

### 1. **NaN Loss Issue** ✅ FIXED
- **Problem**: Training loss was `nan`, causing complete training failure
- **Root Cause**: T5-small FFN structure mismatch and unstable training
- **Solution**: 
  - Fixed T5 FFN structure detection (`wi` instead of `wi_0`)
  - Added conservative learning rate (1e-4 instead of 5e-4)
  - Added NaN gradient detection and skipping
  - Added 0.1 scaling factor for expansion output to start small
  - Added proper dropout and initialization

### 2. **Data Loading Inconsistency** ✅ FIXED
- **Problem**: Different data loading approach than other experiments
- **Original**: Used `load_dataset("code_search_net", "python")` with 8K/2K split
- **Fixed**: Now uses EXACT same approach as LoRA vs Full Layer experiment:
  - `load_dataset("code_search_net", split="train")`
  - Filter by language: `dataset.filter(lambda x: x["language"] == "python")`
  - Same splits: `train=15000, val=5000` (not 8K/2K)

### 3. **Evaluation Sample Size Mismatch** ✅ FIXED
- **Problem**: Used 100 samples for evaluation
- **Fixed**: Now uses 50 samples (same as LoRA vs Full Layer experiment)

### 4. **Model Save/Load Issues** ✅ FIXED
- **Problem**: Custom `ExpandedFFN` modules couldn't be saved/loaded properly
- **Solution**: Implemented custom save/load mechanism:
  - Saves only expansion weights (trainable parameters)
  - Reconstructs model architecture on load
  - Preserves base model + loads expansion weights

### 5. **Training Parameters Alignment** ✅ VERIFIED
- **Epochs**: 2 (same as other experiments)
- **Batch Size**: 8 (same as other experiments)
- **Seed**: 42 (same as other experiments)

## Fair Comparison Checklist

✅ **Same Base Model**: `Salesforce/codet5-small`
✅ **Same Dataset**: CodeSearchNet with identical filtering and splits
✅ **Same Training Data**: Python (15K) → JavaScript (15K)
✅ **Same Validation Data**: Python (5K), JavaScript (5K)
✅ **Same Evaluation**: 50 samples per language
✅ **Same Training Params**: 2 epochs, batch size 8, seed 42
✅ **Same Metrics**: BLEU, Pass Rate, METEOR, Edit Distance, AST Similarity
✅ **Same Experimental Design**: Fresh model per task (no continual learning)

## Key Architecture Details

- **Expansion Size**: 512 (configurable)
- **Trainable Parameters**: ~6.29M (9.42% of model)
- **Architecture**: Parallel expansion paths with residual connections
- **Parameter Efficiency**: More parameters than LoRA (1.78M) but novel approach

## Expected Results Comparison

| Experiment | Parameters | Approach | Expected Performance |
|------------|------------|----------|---------------------|
| LoRA | 1.78M (2.8%) | Low-rank adaptation | Baseline |
| Full Layer | 3.15M (4.9%) | New transformer layers | Higher performance |
| Hybrid | 4.92M (7.7%) | LoRA + Full layer | Best performance |
| **FFN Expansion** | **6.29M (9.42%)** | **Layer widening** | **Novel comparison** |

## Test Results

- ✅ FFN structure correctly identified
- ✅ ExpandedFFN creation and forward pass working
- ✅ Training produces valid loss (not NaN)
- ✅ Model saving/loading working
- ✅ Evaluation metrics computed correctly

The experiment is now ready for fair comparison with existing approaches! 