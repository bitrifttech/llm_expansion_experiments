import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main file
from attention_head_expansion_continual_learning import AttentionHeadExpansionContinualLearner, log_message

def test_mini_experiment():
    """Test a mini version of the full experiment"""
    log_message("Testing mini attention head expansion experiment...")
    
    # Initialize model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cpu"  # Use CPU for testing
    
    # Initialize learner
    learner = AttentionHeadExpansionContinualLearner(model_name, tokenizer, device, num_new_heads=1)
    learner.prepare_model()
    
    log_message("Learner initialized successfully")
    
    # Create mini training data
    mini_train_data = [
        {'input': 'Generate Python code: add two numbers', 'target': 'def add(a, b): return a + b'},
        {'input': 'Generate Python code: multiply two numbers', 'target': 'def multiply(a, b): return a * b'},
        {'input': 'Generate Python code: subtract two numbers', 'target': 'def subtract(a, b): return a - b'},
    ]
    
    # Create mini evaluation data
    mini_eval_data = [
        {'input': 'Generate Python code: divide two numbers', 'target': 'def divide(a, b): return a / b'},
        {'input': 'Generate Python code: find maximum', 'target': 'def max_val(a, b): return max(a, b)'},
    ]
    
    log_message("Created mini datasets")
    
    try:
        # Test training (1 epoch, small batch)
        log_message("Testing training...")
        training_time = learner.train_task(mini_train_data, "python_mini", epochs=1, batch_size=2)
        log_message(f"Training completed in {training_time:.2f} minutes")
        
        # Test evaluation
        log_message("Testing evaluation...")
        bleu_score, pass_rate = learner.evaluate_task(mini_eval_data, "python_mini", num_samples=2)
        log_message(f"Evaluation completed: BLEU={bleu_score:.4f}, Pass Rate={pass_rate:.2f}%")
        
        log_message("SUCCESS: Mini experiment completed without errors!")
        return True
        
    except Exception as e:
        log_message(f"FAILED: Mini experiment failed with error: {e}")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    log_message("Starting mini experiment test...")
    
    if test_mini_experiment():
        log_message("Mini experiment test passed! Full experiment should work.")
    else:
        log_message("Mini experiment test failed!")
        exit(1) 