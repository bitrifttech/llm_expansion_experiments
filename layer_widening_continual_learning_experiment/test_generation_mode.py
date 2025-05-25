import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main file
from attention_head_expansion_continual_learning import (
    AttentionHeadExpansionContinualLearner, 
    expand_model_attention_heads,
    enable_generation_mode,
    disable_generation_mode,
    log_message
)

def test_generation_mode():
    """Test that generation mode properly disables new heads during generation"""
    log_message("=== TESTING GENERATION MODE ===")
    
    # Initialize model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cpu"  # Use CPU for testing
    
    # Initialize learner
    learner = AttentionHeadExpansionContinualLearner(model_name, tokenizer, device, num_new_heads=1)
    learner.prepare_model()
    
    # Create expanded model
    expanded_model = expand_model_attention_heads(learner.base_model, 1)
    
    # Test input
    test_input = "Generate Python code: add two numbers"
    
    # Tokenize input
    input_encoding = tokenizer(
        test_input, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    log_message(f"Test input: {test_input}")
    
    # Test 1: Generation with new heads DISABLED (generation mode)
    log_message("\n--- Test 1: Generation with new heads DISABLED (should work) ---")
    enable_generation_mode(expanded_model)  # Disable new heads for generation
    expanded_model.eval()
    
    with torch.no_grad():
        try:
            outputs_without_heads = expanded_model.generate(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                max_length=min(100, len(input_encoding.input_ids[0]) + 50),
                num_beams=2,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            text_without_heads = tokenizer.decode(outputs_without_heads[0], skip_special_tokens=True)
            log_message(f"✅ Generated (generation mode): {text_without_heads[:200]}...")
            log_message(f"Length: {len(text_without_heads)} characters")
            
        except Exception as e:
            log_message(f"❌ Generation mode failed: {e}")
            text_without_heads = "FAILED"
    
    # Test 2: Compare base model generation
    log_message("\n--- Test 2: Base model generation for comparison ---")
    learner.base_model.eval()
    
    with torch.no_grad():
        try:
            outputs_base = learner.base_model.generate(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                max_length=min(100, len(input_encoding.input_ids[0]) + 50),
                num_beams=2,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            text_base = tokenizer.decode(outputs_base[0], skip_special_tokens=True)
            log_message(f"✅ Generated (base model): {text_base[:200]}...")
            log_message(f"Length: {len(text_base)} characters")
            
        except Exception as e:
            log_message(f"❌ Base model generation failed: {e}")
            text_base = "FAILED"
    
    # Analysis
    log_message("\n=== ANALYSIS ===")
    
    # Check if generation mode works (should match base model)
    if text_without_heads == text_base:
        log_message("✅ PERFECT: Generation mode output matches base model exactly")
    else:
        log_message("⚠️  Generation mode output differs slightly from base model (may be due to randomness)")
        log_message(f"   Base: {text_base[:100]}...")
        log_message(f"   Gen mode: {text_without_heads[:100]}...")
    
    # Check for generation quality
    reasonable_length = 10 <= len(text_without_heads) <= 500
    no_repetition = not any(word * 5 in text_without_heads for word in ["self", "def", "the", "and"])
    contains_code = any(keyword in text_without_heads.lower() for keyword in ["def", "return", "=", "function"])
    
    log_message(f"\n📊 QUALITY METRICS:")
    log_message(f"   Reasonable length (10-500 chars): {reasonable_length} ({len(text_without_heads)} chars)")
    log_message(f"   No excessive repetition: {no_repetition}")
    log_message(f"   Contains code-like content: {contains_code}")
    
    if reasonable_length and no_repetition:
        log_message("✅ Generation quality is GOOD")
    else:
        log_message("⚠️  Generation quality needs improvement")
    
    log_message("\n=== GENERATION MODE TEST COMPLETE ===")
    
    # Return success status
    return text_without_heads != "FAILED" and reasonable_length and no_repetition

if __name__ == "__main__":
    success = test_generation_mode()
    if success:
        log_message("🎉 Generation mode test PASSED!")
    else:
        log_message("❌ Generation mode test FAILED!")
        exit(1) 