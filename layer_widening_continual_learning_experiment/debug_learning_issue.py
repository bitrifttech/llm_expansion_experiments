import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main file
from attention_head_expansion_continual_learning import ExpandedMultiHeadAttention, expand_model_attention_heads, log_message

def debug_learning_issue():
    """Debug why attention heads aren't learning effectively"""
    log_message("=== DEBUGGING ATTENTION HEAD LEARNING ISSUE ===")
    
    # Initialize model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load base model
    base_model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # Create expanded model
    expanded_model = expand_model_attention_heads(base_model, num_new_heads=1)
    
    log_message("Models created successfully")
    
    # Check initial state
    log_message("\n=== INITIAL STATE ANALYSIS ===")
    analyze_attention_state(expanded_model, "INITIAL")
    
    # Create a simple training example
    train_input = "Generate Python code: add two numbers"
    train_target = "def add(a, b): return a + b"
    
    # Tokenize
    input_encoding = tokenizer(
        train_input, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    target_encoding = tokenizer(
        train_target, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    log_message(f"Training example: '{train_input}' -> '{train_target}'")
    
    # Get trainable parameters
    trainable_params = [p for p in expanded_model.parameters() if p.requires_grad]
    log_message(f"Trainable parameters: {len(trainable_params)} groups, {sum(p.numel() for p in trainable_params):,} total")
    
    # Setup optimizer with higher learning rate
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)  # Higher LR
    
    # Training loop with detailed analysis
    expanded_model.train()
    
    log_message("\n=== TRAINING ANALYSIS ===")
    
    for step in range(10):  # 10 training steps
        log_message(f"\n--- STEP {step + 1} ---")
        
        # Forward pass
        outputs = expanded_model(
            input_ids=input_encoding.input_ids,
            attention_mask=input_encoding.attention_mask,
            labels=target_encoding.input_ids
        )
        
        loss = outputs.loss
        log_message(f"Loss: {loss.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Analyze gradients before clipping
        analyze_gradients(expanded_model, step)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Analyze state after update
        if step % 3 == 0:  # Every 3 steps
            analyze_attention_state(expanded_model, f"STEP_{step + 1}")
    
    log_message("\n=== FINAL ANALYSIS ===")
    analyze_attention_state(expanded_model, "FINAL")
    
    # Test inference difference
    log_message("\n=== INFERENCE DIFFERENCE TEST ===")
    test_inference_difference(expanded_model, tokenizer, train_input)

def analyze_attention_state(model, phase):
    """Analyze the state of attention heads"""
    log_message(f"Analyzing attention state - {phase}")
    
    attention_modules = []
    for name, module in model.named_modules():
        if isinstance(module, ExpandedMultiHeadAttention):
            attention_modules.append((name, module))
    
    if not attention_modules:
        log_message("  No ExpandedMultiHeadAttention modules found!")
        return
    
    # Analyze first few modules
    for i, (name, module) in enumerate(attention_modules[:3]):
        stats = module.get_training_stats()
        log_message(f"  {name}:")
        log_message(f"    Gate: {stats['gate_value']:.6f} (sigmoid({module.gate.item():.3f}))")
        log_message(f"    Weight norms: Q={stats['new_q_weight_norm']:.4f}, "
                   f"K={stats['new_k_weight_norm']:.4f}, "
                   f"V={stats['new_v_weight_norm']:.4f}, "
                   f"O={stats['new_o_weight_norm']:.4f}")
        
        # Check if weights are changing
        if hasattr(module, '_prev_weights'):
            q_change = torch.norm(module.new_q_proj.weight.data - module._prev_weights['q']).item()
            k_change = torch.norm(module.new_k_proj.weight.data - module._prev_weights['k']).item()
            v_change = torch.norm(module.new_v_proj.weight.data - module._prev_weights['v']).item()
            o_change = torch.norm(module.new_o_proj.weight.data - module._prev_weights['o']).item()
            log_message(f"    Weight changes: Q={q_change:.6f}, K={k_change:.6f}, "
                       f"V={v_change:.6f}, O={o_change:.6f}")
        
        # Store current weights for next comparison
        module._prev_weights = {
            'q': module.new_q_proj.weight.data.clone(),
            'k': module.new_k_proj.weight.data.clone(),
            'v': module.new_v_proj.weight.data.clone(),
            'o': module.new_o_proj.weight.data.clone()
        }

def analyze_gradients(model, step):
    """Analyze gradients of attention heads"""
    log_message(f"Gradient analysis:")
    
    attention_modules = []
    for name, module in model.named_modules():
        if isinstance(module, ExpandedMultiHeadAttention):
            attention_modules.append((name, module))
    
    if not attention_modules:
        log_message("  No ExpandedMultiHeadAttention modules found!")
        return
    
    # Analyze gradients for first module
    name, module = attention_modules[0]
    
    q_grad = module.new_q_proj.weight.grad
    k_grad = module.new_k_proj.weight.grad
    v_grad = module.new_v_proj.weight.grad
    o_grad = module.new_o_proj.weight.grad
    gate_grad = module.gate.grad
    
    log_message(f"  {name} gradients:")
    log_message(f"    Q: {q_grad.norm().item():.8f} (max: {q_grad.abs().max().item():.8f})")
    log_message(f"    K: {k_grad.norm().item():.8f} (max: {k_grad.abs().max().item():.8f})")
    log_message(f"    V: {v_grad.norm().item():.8f} (max: {v_grad.abs().max().item():.8f})")
    log_message(f"    O: {o_grad.norm().item():.8f} (max: {o_grad.abs().max().item():.8f})")
    log_message(f"    Gate: {gate_grad.item():.8f}")
    
    # Check if gradients are too small
    total_grad_norm = (q_grad.norm() + k_grad.norm() + v_grad.norm() + o_grad.norm()).item()
    if total_grad_norm < 1e-6:
        log_message(f"  WARNING: Very small gradients! Total norm: {total_grad_norm:.10f}")
    
    # Check gradient flow through gate
    gate_contribution = torch.sigmoid(module.gate) * 0.25
    log_message(f"  Gate contribution: {gate_contribution.item():.6f} (max 25%)")

def test_inference_difference(model, tokenizer, test_input):
    """Test if new heads make a difference in inference"""
    log_message("Testing inference difference...")
    
    model.eval()
    
    # Tokenize input
    input_encoding = tokenizer(
        test_input, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        # Test with new heads enabled
        outputs_with = model.generate(
            input_ids=input_encoding.input_ids,
            attention_mask=input_encoding.attention_mask,
            max_length=50,
            num_beams=1,  # Greedy
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        text_with = tokenizer.decode(outputs_with[0], skip_special_tokens=True)
        
        # Test with new heads disabled (set gates to very negative)
        original_gates = []
        for name, module in model.named_modules():
            if isinstance(module, ExpandedMultiHeadAttention):
                original_gates.append(module.gate.data.clone())
                module.gate.data.fill_(-50.0)  # Very negative = ~0 contribution
        
        outputs_without = model.generate(
            input_ids=input_encoding.input_ids,
            attention_mask=input_encoding.attention_mask,
            max_length=50,
            num_beams=1,  # Greedy
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        text_without = tokenizer.decode(outputs_without[0], skip_special_tokens=True)
        
        # Restore gates
        gate_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, ExpandedMultiHeadAttention):
                module.gate.data = original_gates[gate_idx]
                gate_idx += 1
    
    log_message(f"  With new heads: '{text_with}'")
    log_message(f"  Without new heads: '{text_without}'")
    log_message(f"  Different outputs: {text_with != text_without}")
    
    if text_with == text_without:
        log_message("  ISSUE: New heads are not affecting inference!")
    else:
        log_message("  SUCCESS: New heads are affecting inference!")

if __name__ == "__main__":
    log_message("Starting attention head learning debug...")
    debug_learning_issue()
    log_message("Debug completed!") 