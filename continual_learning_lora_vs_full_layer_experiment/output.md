python lora_vs_full_layer_training_mac_and_cuda.py

[2025-05-24 13:44:36] [INFO] Using CUDA GPU
[2025-05-24 13:44:36] [INFO] GPU: NVIDIA GeForce RTX 4060 Ti, Memory: 15.67 GB
[2025-05-24 13:44:36] [INFO] Device: cuda, System Memory: 94.22 GB
[2025-05-24 13:44:36] [INFO] Starting LoRA vs New Layer Training Comparison
[2025-05-24 13:44:36] [INFO] FAIR COMPARISON: Both approaches use task-specific component swapping
[2025-05-24 13:44:36] [INFO] LoRA: Frozen base + swappable adapter matrices (~0.1% parameters)
[2025-05-24 13:44:36] [INFO] New Layer: Frozen base + swappable transformer layers (~5% parameters)
[2025-05-24 13:44:36] [INFO] Both: Perfect task isolation, no cross-task interference during training
[2025-05-24 13:44:36] [INFO] Tokenizer loaded successfully
[2025-05-24 13:44:36] [INFO] Loading CodeSearchNet dataset...
[2025-05-24 13:44:37] [INFO] Dataset prepared: Python train=15000, val=5000
[2025-05-24 13:44:37] [INFO]                   JavaScript train=15000, val=5000
[2025-05-24 13:44:37] [INFO] 
============================================================
[2025-05-24 13:44:37] [INFO] EXPERIMENTAL TRIAL 1/1 (seed: 42)
[2025-05-24 13:44:37] [INFO] ============================================================
[2025-05-24 13:44:37] [INFO] Running LoRAContinualLearner experiment with seed 42
[2025-05-24 13:44:38] [INFO] === BASELINE EVALUATION (Untrained Model) ===
Token indices sequence length is longer than the specified maximum sequence length for this model (525 > 512). Running this sequence through the model will result in indexing errors
[2025-05-24 13:45:12] [INFO] Baseline Python - BLEU: 0.0038, METEOR: 0.0069, AST: 0.0483
[2025-05-24 13:45:12] [INFO] Baseline JavaScript - BLEU: 0.0196, METEOR: 0.0107, AST: 0.3344
[2025-05-24 13:45:12] [INFO] 
=== STEP 1: TRAINING ON PYTHON ===
[2025-05-24 13:45:12] [INFO] Training LoRA adapter for python...
Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.
[2025-05-24 13:50:57] [INFO] Epoch 1/2, Avg Loss: 0.0643
[2025-05-24 13:56:42] [INFO] Epoch 2/2, Avg Loss: 0.0105
[2025-05-24 13:56:43] [INFO] LoRA adapter for python saved to adapters/python
[2025-05-24 13:56:43] [INFO] Comprehensive evaluation after Python training...
[2025-05-24 13:56:43] [INFO] Successfully switched to LoRA adapter for python (device: cuda:0)
[2025-05-24 13:56:43] [INFO] Current model for Python evaluation: PeftModelForSeq2SeqLM
[2025-05-24 13:57:39] [INFO] After Python training:
[2025-05-24 13:57:39] [INFO]   Python BLEU: 0.0038 → 0.2166 (Δ+0.2128)
[2025-05-24 13:57:39] [INFO]   Python METEOR: 0.0069 → 0.0445 (Δ+0.0376)
[2025-05-24 13:57:39] [INFO]   Python AST Similarity: 0.0483 → 0.0577
[2025-05-24 13:57:39] [INFO]   Python Code Complexity: 3.32
[2025-05-24 13:57:39] [INFO] 
=== STEP 2: TRAINING ON JAVASCRIPT ===
[2025-05-24 13:57:39] [INFO] Training LoRA adapter for javascript...
[2025-05-24 14:03:23] [INFO] Epoch 1/2, Avg Loss: 0.0591
[2025-05-24 14:09:08] [INFO] Epoch 2/2, Avg Loss: 0.0105
[2025-05-24 14:09:08] [INFO] LoRA adapter for javascript saved to adapters/javascript
[2025-05-24 14:09:08] [INFO] Comprehensive evaluation after JavaScript training...
[2025-05-24 14:09:08] [INFO] Successfully switched to LoRA adapter for javascript (device: cuda:0)
[2025-05-24 14:09:08] [INFO] Current model for JavaScript evaluation: PeftModelForSeq2SeqLM
[2025-05-24 14:09:47] [INFO] Successfully switched to LoRA adapter for python (device: cuda:0)
[2025-05-24 14:09:47] [INFO] Current model for Python re-evaluation: PeftModelForSeq2SeqLM
[2025-05-24 14:10:44] [INFO] After JavaScript training:
[2025-05-24 14:10:44] [INFO]   JavaScript BLEU: 0.0196 → 0.2489 (Δ+0.2292)
[2025-05-24 14:10:44] [INFO]   JavaScript METEOR: 0.0107 → 0.0000
[2025-05-24 14:10:44] [INFO]   JavaScript AST Similarity: 0.6268
[2025-05-24 14:10:44] [INFO]   Python BLEU: 0.2166 → 0.2150 (Δ-0.0017)
[2025-05-24 14:10:44] [INFO]   Python Forgetting: 0.8%
[2025-05-24 14:10:44] [INFO] 
=== CONTINUAL LEARNING ANALYSIS ===
[2025-05-24 14:10:44] [INFO] 🔄 Forward Transfer: 0.0000
[2025-05-24 14:10:44] [INFO] 🔙 Backward Interference: 0.0017
[2025-05-24 14:10:44] [INFO] 🧠 Retention Score: 6.8297
[2025-05-24 14:10:44] [INFO] 
=== COMPREHENSIVE SUMMARY ===
[2025-05-24 14:10:44] [INFO] 📊 Final Python Performance: BLEU 0.2150, METEOR 0.0531
[2025-05-24 14:10:44] [INFO] 📊 Final JavaScript Performance: BLEU 0.2489, METEOR 0.0000
[2025-05-24 14:10:44] [INFO] ⚡ Training Efficiency: 22.97 min, 1.77 GB
[2025-05-24 14:10:44] [INFO] 
LoRAContinualLearner Comprehensive Results (seed 42):
[2025-05-24 14:10:44] [INFO]   📊 Performance: Python BLEU 0.2150, JS BLEU 0.2489
[2025-05-24 14:10:44] [INFO]   🧠 Continual Learning: Forgetting 0.78%, Retention 6.8297
[2025-05-24 14:10:44] [INFO]   ⚡ Efficiency: 22.97 min, 1.77 GB
[2025-05-24 14:10:44] [INFO] Running FullLayerContinualLearner experiment with seed 42
[2025-05-24 14:10:45] [INFO] Froze 131 base model parameters
[2025-05-24 14:10:45] [INFO] Base model prepared with frozen weights
[2025-05-24 14:10:45] [INFO] === BASELINE EVALUATION (Untrained Model) ===
[2025-05-24 14:11:19] [INFO] Baseline Python - BLEU: 0.0038, METEOR: 0.0069, AST: 0.0483
[2025-05-24 14:11:19] [INFO] Baseline JavaScript - BLEU: 0.0196, METEOR: 0.0107, AST: 0.3344
[2025-05-24 14:11:19] [INFO] 
=== STEP 1: TRAINING ON PYTHON ===
[2025-05-24 14:11:19] [INFO] Training new transformer layer for python...
[2025-05-24 14:11:20] [INFO] Created new model with additional layer: 3,146,752 trainable / 63,639,040 total parameters
[2025-05-24 14:11:20] [INFO] Training 3,146,752 / 63,639,040 parameters (4.94%)
[2025-05-24 14:16:45] [INFO] Epoch 1/2, Avg Loss: 0.6641
[2025-05-24 14:22:11] [INFO] Epoch 2/2, Avg Loss: 0.2955
[2025-05-24 14:22:12] [INFO] Task-specific checkpoint for python saved to checkpoints/python
[2025-05-24 14:22:12] [INFO] Model architecture: Base (frozen) + python Layer (trained)
[2025-05-24 14:22:12] [INFO] Comprehensive evaluation after Python training...
[2025-05-24 14:22:12] [INFO] Successfully switched to checkpoint for python (device: cuda:0)
[2025-05-24 14:22:12] [INFO] Current model for Python evaluation: T5ForConditionalGeneration
[2025-05-24 14:22:44] [INFO] After Python training:
[2025-05-24 14:22:44] [INFO]   Python BLEU: 0.0038 → 0.1927 (Δ+0.1888)
[2025-05-24 14:22:44] [INFO]   Python METEOR: 0.0069 → 0.0138 (Δ+0.0069)
[2025-05-24 14:22:44] [INFO]   Python AST Similarity: 0.0483 → 0.0310
[2025-05-24 14:22:44] [INFO]   Python Code Complexity: 3.40
[2025-05-24 14:22:44] [INFO] 
=== STEP 2: TRAINING ON JAVASCRIPT ===
[2025-05-24 14:22:44] [INFO] Training new transformer layer for javascript...
[2025-05-24 14:22:45] [INFO] Created new model with additional layer: 3,146,752 trainable / 63,639,040 total parameters
[2025-05-24 14:22:45] [INFO] Training 3,146,752 / 63,639,040 parameters (4.94%)
[2025-05-24 14:28:09] [INFO] Epoch 1/2, Avg Loss: 0.5956
[2025-05-24 14:33:34] [INFO] Epoch 2/2, Avg Loss: 0.2726
[2025-05-24 14:33:35] [INFO] Task-specific checkpoint for javascript saved to checkpoints/javascript
[2025-05-24 14:33:35] [INFO] Model architecture: Base (frozen) + javascript Layer (trained)
[2025-05-24 14:33:35] [INFO] Comprehensive evaluation after JavaScript training...
[2025-05-24 14:33:35] [INFO] Successfully switched to checkpoint for javascript (device: cuda:0)
[2025-05-24 14:33:35] [INFO] Current model for JavaScript evaluation: T5ForConditionalGeneration
[2025-05-24 14:33:59] [INFO] Successfully switched to checkpoint for python (device: cuda:0)
[2025-05-24 14:33:59] [INFO] Current model for Python re-evaluation: T5ForConditionalGeneration
[2025-05-24 14:34:33] [INFO] After JavaScript training:
[2025-05-24 14:34:33] [INFO]   JavaScript BLEU: 0.0196 → 0.2147 (Δ+0.1950)
[2025-05-24 14:34:33] [INFO]   JavaScript METEOR: 0.0107 → 0.0000
[2025-05-24 14:34:33] [INFO]   JavaScript AST Similarity: 0.5978
[2025-05-24 14:34:33] [INFO]   Python BLEU: 0.1927 → 0.2046 (Δ+0.0120)
[2025-05-24 14:34:33] [INFO]   Python Forgetting: -6.2%
[2025-05-24 14:34:33] [INFO] 
=== CONTINUAL LEARNING ANALYSIS ===
[2025-05-24 14:34:33] [INFO] 🔄 Forward Transfer: 0.0000
[2025-05-24 14:34:33] [INFO] 🔙 Backward Interference: 0.0000
[2025-05-24 14:34:33] [INFO] 🧠 Retention Score: 5.9948
[2025-05-24 14:34:33] [INFO] 
=== COMPREHENSIVE SUMMARY ===
[2025-05-24 14:34:33] [INFO] 📊 Final Python Performance: BLEU 0.2046, METEOR 0.0138
[2025-05-24 14:34:33] [INFO] 📊 Final JavaScript Performance: BLEU 0.2147, METEOR 0.0000
[2025-05-24 14:34:33] [INFO] ⚡ Training Efficiency: 21.69 min, 0.01 GB
[2025-05-24 14:34:33] [INFO] 
FullLayerContinualLearner Comprehensive Results (seed 42):
[2025-05-24 14:34:33] [INFO]   📊 Performance: Python BLEU 0.2046, JS BLEU 0.2147
[2025-05-24 14:34:33] [INFO]   🧠 Continual Learning: Forgetting -6.20%, Retention 5.9948
[2025-05-24 14:34:33] [INFO]   ⚡ Efficiency: 21.69 min, 0.01 GB
[2025-05-24 14:34:33] [WARNING] Insufficient successful experiments for statistical analysis
[2025-05-24 14:34:33] [INFO] Experiment completed!