import os
import sys
import torch
import numpy as np
import psutil
import time
from copy import deepcopy
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from peft import LoraConfig, get_peft_model, PeftModel
import shutil

# Logging setup
def log_message(message, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

# Debug imports
log_message("Starting script...")
for module in [torch, np, psutil, time, deepcopy, sentence_bleu, load_dataset, T5ForConditionalGeneration, T5Tokenizer, T5Config, LoraConfig, get_peft_model, PeftModel]:
    log_message(f"Imported {module.__name__} successfully")

# Device setup (MPS, CUDA, or CPU)
if torch.cuda.is_available():
    device = "cuda"
    log_message("Using CUDA GPU")
    log_message(f"GPU: {torch.cuda.get_device_name()}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
elif torch.backends.mps.is_available():
    device = "mps"
    log_message("Using Apple Silicon MPS")
else:
    device = "cpu"
    log_message("Using CPU (no MPS or CUDA available)")
log_message(f"Device: {device}, System Memory: {psutil.virtual_memory().total / 1024**2:.2f} MB")

# 1. Load Dataset
try:
    log_message("Loading CodeSearchNet dataset...")
    dataset = load_dataset("code_search_net", split="train")
    key = "whole_func_string" if "whole_func_string" in dataset[0] else "func_code_string"
    python_data = dataset.filter(lambda x: x["language"] == "python").select(range(10000))
    js_data = dataset.filter(lambda x: x["language"] == "javascript").select(range(10000))
    python_val = dataset.filter(lambda x: x["language"] == "python").select(range(10000, 11000))
    js_val = dataset.filter(lambda x: x["language"] == "javascript").select(range(10000, 11000))
    log_message(f"Dataset sizes: Python {len(python_data)}, JavaScript {len(js_data)}, Python Val {len(python_val)}, JavaScript Val {len(js_val)}")
    log_message(f"Using key: {key}")
    # Validate dataset structure
    log_message(f"Sample Python data: {python_data[0][key][:50]}...")
    if not isinstance(python_data[0], dict) or key not in python_data[0]:
        raise ValueError(f"Dataset does not contain dictionaries with key '{key}'")
except Exception as e:
    log_message(f"Dataset error: {e}", level="ERROR")
    sys.exit(1)

# 2. Initialize Model and Tokenizer
model_name = "Salesforce/codet5-small"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Salesforce--codet5-small")
try:
    log_message(f"Clearing cache at {cache_dir}...")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        log_message("Cache cleared successfully")
except Exception as e:
    log_message(f"Cache clearing error: {e}", level="WARNING")

try:
    log_message("Loading T5Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    if not isinstance(tokenizer, T5Tokenizer):
        raise ValueError(f"Loaded tokenizer is {type(tokenizer).__name__}, expected T5Tokenizer")
    log_message("T5Tokenizer loaded successfully")
except Exception as e:
    log_message(f"T5Tokenizer error: {e}. Trying t5-small as fallback.", level="ERROR")
    try:
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        log_message("Fallback to t5-small tokenizer succeeded, but may affect performance", level="WARNING")
    except Exception as e2:
        log_message(f"Fallback tokenizer error: {e2}. Exiting.", level="ERROR")
        sys.exit(1)

try:
    log_message("Loading T5Config...")
    config = T5Config.from_pretrained(model_name)
except Exception as e:
    log_message(f"Config error: {e}", level="ERROR")
    sys.exit(1)

# 3. Add Transformer Layer
def add_transformer_layer(model, config):
    try:
        log_message("Adding new transformer layer...")
        
        # Create a deep copy of the last layer
        new_layer = deepcopy(model.encoder.block[-1])
        
        # Add the new layer to the encoder
        model.encoder.block.append(new_layer)
        
        # Update the configuration
        config.num_layers = len(model.encoder.block)
        
        # CRITICAL: Update the model's config to match the new architecture
        model.config.num_layers = config.num_layers
        
        log_message(f"Added transformer layer: {config.num_layers-1} -> {config.num_layers} layers")
        
        # Initialize the new layer parameters with small random values
        for param in new_layer.parameters():
            param.data = param.data * 0.01
            
        # MONKEY PATCH: Fix the head_mask issue
        original_get_head_mask = model.encoder.get_head_mask
        
        def patched_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            # Always use the actual number of encoder blocks
            actual_num_layers = len(self.block)
            result = original_get_head_mask(head_mask, actual_num_layers, is_attention_chunked)
            return result
        
        # Apply the monkey patch
        import types
        model.encoder.get_head_mask = types.MethodType(patched_get_head_mask, model.encoder)
            
        # Force model to reinitialize internal structures by moving to device
        device = next(model.parameters()).device
        model = model.to(device)
        
        return model
    except Exception as e:
        log_message(f"Layer addition error: {e}", level="ERROR")
        raise

# 4. Training Function
def train_model(model, data, replay_data, key, epochs=1, batch_size=2, accum_steps=4):
    try:
        log_message(f"Starting training: epochs={epochs}, batch_size={batch_size}, accum_steps={accum_steps}")
        start_time = time.time()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        model.train()
        losses = []
        batch_times = []
        total_batches = (len(data) // batch_size) * epochs
        if replay_data:
            total_batches += (len(replay_data) // batch_size) * epochs
        batches_processed = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_losses = []
            log_message(f"Starting epoch {epoch+1}/{epochs}")

            # Main data
            for i in range(0, len(data), batch_size):
                batch_start = time.time()
                # Handle both Dataset and dict objects
                if hasattr(data, 'select'):
                    # Use .select() for datasets
                    batch_indices = list(range(i, min(i + batch_size, len(data))))
                    batch_data = data.select(batch_indices)
                else:
                    # Handle dict objects (fallback)
                    end_idx = min(i + batch_size, len(data))
                    batch_data = {k: v[i:end_idx] for k, v in data.items()}
                    
                batch = [text for text in batch_data[key] if text and str(text).strip()]
                
                if not batch:  # Skip empty batches
                    continue
                    
                inputs = tokenizer(batch, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
                
                # Forward pass
                try:
                    outputs = model(**inputs, labels=inputs["input_ids"])
                except Exception as e:
                    log_message(f"CRITICAL ERROR during model forward pass: {e}", level="ERROR")
                    log_message(f"Exception type: {type(e).__name__}", level="ERROR")
                    import traceback
                    traceback.print_exc()
                    
                    # Dump relevant state for debugging
                    log_message("=== DEBUGGING STATE DUMP ===", level="ERROR")
                    log_message(f"Batch size: {len(batch)}", level="ERROR")
                    log_message(f"Input IDs shape: {inputs['input_ids'].shape}", level="ERROR")
                    log_message(f"Model config: {model.config}", level="ERROR")
                    log_message(f"Current memory usage: {get_memory_usage():.2f} MB", level="ERROR")
                    raise

                loss = outputs.loss / accum_steps
                loss.backward()
                if (i // batch_size + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                epoch_losses.append(loss.item() * accum_steps)

                # Progress logging (every 100 batches to reduce spam)
                batches_processed += 1
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                if batches_processed % 100 == 0 or batches_processed <= 10:
                    avg_batch_time = np.mean(batch_times)
                    remaining_batches = total_batches - batches_processed
                    eta_seconds = remaining_batches * avg_batch_time
                    eta_min = eta_seconds / 60
                    log_message(f"Epoch {epoch+1}, Batch {i//batch_size+1}/{len(data)//batch_size}, Loss: {loss.item():.4f}, Memory: {get_memory_usage():.2f} MB, ETA: {eta_min:.2f} min")

            # Replay data
            if replay_data:
                log_message("Processing replay data...")
                for i in range(0, len(replay_data), batch_size):
                    batch_start = time.time()
                    # Handle both Dataset and dict objects
                    if hasattr(replay_data, 'select'):
                        # Use .select() for datasets
                        batch_indices = list(range(i, min(i + batch_size, len(replay_data))))
                        batch_data = replay_data.select(batch_indices)
                    else:
                        # Handle dict objects (fallback)
                        end_idx = min(i + batch_size, len(replay_data))
                        batch_data = {k: v[i:end_idx] for k, v in replay_data.items()}
                    
                    batch = [text for text in batch_data[key] if text and str(text).strip()]
                    if not batch:  # Skip empty batches
                        continue
                    inputs = tokenizer(batch, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
                    
                    # Forward pass
                    try:
                        outputs = model(**inputs, labels=inputs["input_ids"])
                    except Exception as e:
                        log_message(f"CRITICAL ERROR during model forward pass: {e}", level="ERROR")
                        log_message(f"Exception type: {type(e).__name__}", level="ERROR")
                        import traceback
                        traceback.print_exc()
                        
                        # Dump relevant state for debugging
                        log_message("=== DEBUGGING STATE DUMP ===", level="ERROR")
                        log_message(f"Batch size: {len(batch)}", level="ERROR")
                        log_message(f"Input IDs shape: {inputs['input_ids'].shape}", level="ERROR")
                        log_message(f"Model config: {model.config}", level="ERROR")
                        log_message(f"Current memory usage: {get_memory_usage():.2f} MB", level="ERROR")
                        raise

                    loss = outputs.loss / accum_steps
                    loss.backward()
                    if (i // batch_size + 1) % accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    epoch_losses.append(loss.item() * accum_steps)

                    batches_processed += 1
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    
                    if batches_processed % 100 == 0:
                        avg_batch_time = np.mean(batch_times)
                        remaining_batches = total_batches - batches_processed
                        eta_seconds = remaining_batches * avg_batch_time
                        eta_min = eta_seconds / 60
                        log_message(f"Epoch {epoch+1}, Replay Batch {i//batch_size+1}/{len(replay_data)//batch_size}, Loss: {loss.item():.4f}, Memory: {get_memory_usage():.2f} MB, ETA: {eta_min:.2f} min")

            # Epoch summary
            epoch_time = (time.time() - epoch_start) / 60
            log_message(f"Epoch {epoch+1} completed. Avg Loss: {np.mean(epoch_losses):.4f}, Time: {epoch_time:.2f} min, Memory: {get_memory_usage():.2f} MB")
            losses.extend(epoch_losses)

        training_time = (time.time() - start_time) / 60
        log_message(f"Training completed. Total Time: {training_time:.2f} min")
        return np.mean(losses), training_time
    except Exception as e:
        log_message(f"Training error: {e}", level="ERROR")
        raise

# 5. Evaluation Function
def evaluate_model(model, val_data, key):
    try:
        log_message("Starting evaluation...")
        model.eval()
        bleu_scores = []
        pass_scores = []
        eval_start = time.time()
        # Get the number of samples from the length of any key's list
        num_samples = len(val_data[key])
        log_message(f"Evaluating on {min(10, num_samples)} samples...")
        
        for i in range(min(10, num_samples)):  # Only process first 10 samples for debugging
            input_code = val_data[key][i][:64]
            target = val_data[key][i]
            
            inputs = tokenizer(input_code, return_tensors="pt", max_length=128, truncation=True).to(device)
            outputs = model.generate(**inputs, max_length=128)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if not pred.strip() or not target.strip():
                bleu_scores.append(0.0)
                pass_scores.append(0.0)
                continue
            target_tokens = tokenizer.tokenize(target)
            pred_tokens = tokenizer.tokenize(pred)
            try:
                bleu = sentence_bleu([target_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
            except:
                bleu = 0.0
            bleu_scores.append(bleu)
            try:
                if "python" in key.lower():
                    compile(pred, "<string>", "exec")
                else:  # JavaScript
                    eval(f"({pred})")
                pass_scores.append(1.0)
            except:
                pass_scores.append(0.0)
                
        eval_time = (time.time() - eval_start) / 60
        log_message(f"Evaluation completed in {eval_time:.2f} min")
        return np.mean(bleu_scores) if bleu_scores else 0.0, np.mean(pass_scores) if pass_scores else 0.0
    except Exception as e:
        log_message(f"Evaluation error: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        raise

# 6. Memory Usage
def get_memory_usage():
    return psutil.Process().memory_info().rss / 1024**2

# 7. Experiment: LoRA (Attention-Only)
print("=== LoRA (Attention-Only) ===")
try:
    log_message("Loading LoRA model...")
    model_lora = T5ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    ).to(device)
    
    model_lora = add_transformer_layer(model_lora, config)
    
    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q", "k", "v", "o"], task_type="SEQ_2_SEQ_LM"
    )
    
    model_lora = get_peft_model(model_lora, lora_config)
    log_message("LoRA model created successfully")
    
    log_message("Training LoRA on Python...")
    python_loss_lora, python_time_lora = train_model(model_lora, python_data, replay_data=None, key=key)
    log_message("Evaluating LoRA on Python...")
    python_bleu_before_lora, python_pass_before_lora = evaluate_model(model_lora, python_val, key)
    memory_lora_train = get_memory_usage()
    log_message(f"Python BLEU (before): {python_bleu_before_lora:.4f}, Pass@1: {python_pass_before_lora:.4f}, Loss: {python_loss_lora:.4f}, Time: {python_time_lora:.2f} min")
    log_message(f"Training Memory: {memory_lora_train:.2f} MB")
    os.makedirs("lora_python", exist_ok=True)
    model_lora.save_pretrained("lora_python")

    log_message("Training LoRA on JavaScript...")
    model_lora = add_transformer_layer(model_lora, config)
    model_lora = get_peft_model(model_lora, lora_config)
    js_loss_lora, js_time_lora = train_model(model_lora, js_data, replay_data=python_data.select(range(1000)), key=key)
    os.makedirs("lora_javascript", exist_ok=True)
    model_lora.save_pretrained("lora_javascript")
    log_message("Evaluating LoRA on JavaScript...")
    js_bleu_lora, js_pass_lora = evaluate_model(PeftModel.from_pretrained(model_lora, "lora_javascript").to(device), js_val, key)
    log_message("Evaluating LoRA on Python (post-JavaScript)...")
    python_bleu_after_lora, python_pass_after_lora = evaluate_model(PeftModel.from_pretrained(model_lora, "lora_python").to(device), python_val, key)
    memory_lora_infer = get_memory_usage()
    log_message(f"JavaScript BLEU: {js_bleu_lora:.4f}, Pass@1: {js_pass_lora:.4f}, Loss: {js_loss_lora:.4f}, Time: {js_time_lora:.2f} min")
    log_message(f"Python BLEU (after): {python_bleu_after_lora:.4f}, Pass@1: {python_pass_after_lora:.4f}, Forgetting: {(python_bleu_before_lora - python_bleu_after_lora):.4f}")
    log_message(f"Inference Memory: {memory_lora_infer:.2f} MB")
except Exception as e:
    log_message(f"LoRA experiment error: {e}", level="ERROR")
    raise

# 8. Experiment: Full Layer Training
print("\n=== Full Layer Training ===")
try:
    log_message("Loading full layer model...")
    model_full = T5ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    ).to(device)
    model_full = add_transformer_layer(model_full, config)
    log_message("Training full layer on Python...")
    python_loss_full, python_time_full = train_model(model_full, python_data, replay_data=None, key=key)
    log_message("Evaluating full layer on Python...")
    python_bleu_before_full, python_pass_before_full = evaluate_model(model_full, python_val, key)
    memory_full_train = get_memory_usage()
    log_message(f"Python BLEU (before): {python_bleu_before_full:.4f}, Pass@1: {python_pass_before_full:.4f}, Loss: {python_loss_full:.4f}, Time: {python_time_full:.2f} min")
    log_message(f"Training Memory: {memory_full_train:.2f} MB")
    os.makedirs("full_python", exist_ok=True)
    model_full.save_pretrained("full_python")

    log_message("Training full layer on JavaScript...")
    model_full = add_transformer_layer(model_full, config)
    js_loss_full, js_time_full = train_model(model_full, js_data, replay_data=python_data.select(range(1000)), key=key)
    log_message("Evaluating full layer on JavaScript...")
    js_bleu_full, js_pass_full = evaluate_model(model_full, js_val, key)
    os.makedirs("full_javascript", exist_ok=True)
    model_full.save_pretrained("full_javascript")
    log_message("Evaluating full layer on Python (post-JavaScript)...")
    python_bleu_after_full, python_pass_after_full = evaluate_model(T5ForConditionalGeneration.from_pretrained("full_python").to(device), python_val, key)
    memory_full_infer = get_memory_usage()
    log_message(f"JavaScript BLEU: {js_bleu_full:.4f}, Pass@1: {js_pass_full:.4f}, Loss: {js_loss_full:.4f}, Time: {js_time_full:.2f} min")
    log_message(f"Python BLEU (after): {python_bleu_after_full:.4f}, Pass@1: {python_pass_after_full:.4f}, Forgetting: {(python_bleu_before_full - python_bleu_after_full):.4f}")
    log_message(f"Inference Memory: {memory_full_infer:.2f} MB")
except Exception as e:
    log_message(f"Full layer experiment error: {e}", level="ERROR")
    raise