import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main file
from attention_head_expansion_continual_learning import AttentionHeadExpansionContinualLearner, log_message

def test_generation_fix():
    """Test that the generation fix prevents repetitive output"""
    log_message("=== TESTING GENERATION FIX ===")
    
    # Initialize model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cpu"  # Use CPU for testing
    
    # Initialize learner
    learner = AttentionHeadExpansionContinualLearner(model_name, tokenizer, device, num_new_heads=1)
    learner.prepare_model()
    
    # Create expanded model
    from attention_head_expansion_continual_learning import expand_model_attention_heads
    expanded_model = expand_model_attention_heads(learner.base_model, 1)
    
    # Test inputs that were causing issues
    test_inputs = [
        "Generate Python code: add two numbers",
        "Generate Python code: Transform a sequence to internal indexing",
        "Generate Python code: create a simple function"
    ]
    
    expanded_model.eval()
    
    log_message("Testing generation with fixed parameters...")
    
    for i, input_text in enumerate(test_inputs):
        log_message(f"\n--- Test {i+1} ---")
        log_message(f"Input: {input_text}")
        
        # Tokenize input
        input_encoding = tokenizer(
            input_text, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        try:
            # Generate with fixed parameters
            with torch.no_grad():
                outputs = expanded_model.generate(
                    input_ids=input_encoding.input_ids,
                    attention_mask=input_encoding.attention_mask,
                    max_length=min(256, len(input_encoding.input_ids[0]) + 100),
                    num_beams=2,  # Reduced for faster testing
                    early_stopping=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode prediction
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            log_message(f"Generated: {predicted_text[:200]}...")
            log_message(f"Length: {len(predicted_text)} characters")
            
            # Check for repetitive patterns
            words = predicted_text.split()
            if len(words) > 10:
                # Check for excessive repetition
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                max_repetition = max(word_counts.values()) if word_counts else 0
                repetition_ratio = max_repetition / len(words) if words else 0
                
                log_message(f"Max word repetition: {max_repetition}/{len(words)} ({repetition_ratio:.2%})")
                
                if repetition_ratio > 0.3:  # More than 30% repetition
                    log_message("⚠️  WARNING: High repetition detected!")
                else:
                    log_message("✅ Repetition within acceptable range")
            
            # Check for reasonable length
            if len(predicted_text) > 1000:
                log_message("⚠️  WARNING: Generated text is very long!")
            else:
                log_message("✅ Generated length is reasonable")
                
        except Exception as e:
            log_message(f"❌ Generation failed: {e}")
    
    log_message("\n=== GENERATION TEST COMPLETE ===")

if __name__ == "__main__":
    test_generation_fix() 