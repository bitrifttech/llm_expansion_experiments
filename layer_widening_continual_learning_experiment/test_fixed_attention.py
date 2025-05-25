import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the fixed attention class from the main file
from attention_head_expansion_continual_learning import ExpandedMultiHeadAttention, log_message

def test_fixed_attention_generation():
    """Test the fixed attention mechanism with generation"""
    log_message("Testing FIXED attention mechanism with generation...")
    
    # Load model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # Replace one attention layer with our fixed version
    original_attention = model.encoder.block[0].layer[0].SelfAttention
    fixed_attention = ExpandedMultiHeadAttention(original_attention, num_new_heads=1, device="cpu")
    model.encoder.block[0].layer[0].SelfAttention = fixed_attention
    
    log_message("Replaced first encoder attention layer with FIXED version")
    
    # Test with a simple input
    test_input = "Generate Python code: add two numbers"
    
    log_message(f"Testing with input: {test_input}")
    
    # Tokenize
    input_encoding = tokenizer(
        test_input, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    log_message(f"Input encoding shape: {input_encoding.input_ids.shape}")
    
    try:
        # Test beam search generation (the problematic case)
        log_message("Testing beam search generation...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                max_length=100,  # Reasonable length
                num_beams=4,     # Full beam search
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log_message(f"Generated text: {generated_text}")
        log_message("SUCCESS: Beam search generation completed without errors!")
        return True
        
    except Exception as e:
        log_message(f"FAILED: Generation failed with error: {e}")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")
        return False

def test_multiple_samples():
    """Test with multiple samples to ensure robustness"""
    log_message("Testing with multiple samples...")
    
    # Load model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # Replace one attention layer with our fixed version
    original_attention = model.encoder.block[0].layer[0].SelfAttention
    fixed_attention = ExpandedMultiHeadAttention(original_attention, num_new_heads=1, device="cpu")
    model.encoder.block[0].layer[0].SelfAttention = fixed_attention
    
    test_inputs = [
        "Generate Python code: hello world",
        "Generate JavaScript code: function to multiply",
        "Create a function that returns the sum",
        "Write code to sort a list",
        "Implement a simple calculator"
    ]
    
    success_count = 0
    
    for i, test_input in enumerate(test_inputs):
        log_message(f"Testing sample {i+1}: {test_input}")
        
        # Tokenize
        input_encoding = tokenizer(
            test_input, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        try:
            # Test generation
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_encoding.input_ids,
                    attention_mask=input_encoding.attention_mask,
                    max_length=50,   # Shorter for testing
                    num_beams=2,     # Smaller beam size
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            log_message(f"  SUCCESS: {generated_text[:50]}...")
            success_count += 1
            
        except Exception as e:
            log_message(f"  FAILED: {e}")
    
    log_message(f"Multiple samples test: {success_count}/{len(test_inputs)} successful")
    return success_count == len(test_inputs)

if __name__ == "__main__":
    log_message("Starting FIXED attention mechanism tests...")
    
    # Test 1: Basic generation
    if not test_fixed_attention_generation():
        log_message("Basic generation test failed!")
        exit(1)
    
    # Test 2: Multiple samples
    if not test_multiple_samples():
        log_message("Multiple samples test failed!")
        exit(1)
    
    log_message("All tests passed! The fix works correctly.") 