import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffn_expansion_continual_learning import (
    ExpandedFFN, expand_model_ffn, FFNExpansionContinualLearner,
    log_message, device, set_seed
)

def test_ffn_structure():
    """Test that we correctly identify T5 FFN structure"""
    log_message("Testing T5 FFN structure identification...")
    
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")
    ffn = model.encoder.block[0].layer[1].DenseReluDense
    
    log_message(f"FFN has 'wi': {hasattr(ffn, 'wi')}")
    log_message(f"FFN has 'wi_0': {hasattr(ffn, 'wi_0')}")
    
    if hasattr(ffn, 'wi'):
        log_message(f"wi shape: {ffn.wi.weight.shape}")
        log_message(f"wo shape: {ffn.wo.weight.shape}")
    
    return True

def test_expanded_ffn_creation():
    """Test ExpandedFFN creation and forward pass"""
    log_message("Testing ExpandedFFN creation...")
    
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small").to(device)
    original_ffn = model.encoder.block[0].layer[1].DenseReluDense
    
    # Create expanded FFN
    expanded_ffn = ExpandedFFN(original_ffn, expansion_size=256, device=device)
    
    # Test forward pass
    batch_size, seq_len, d_model = 2, 10, 512
    test_input = torch.randn(batch_size, seq_len, d_model, dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    
    with torch.no_grad():
        original_output = original_ffn(test_input)
        expanded_output = expanded_ffn(test_input)
    
    log_message(f"Original output shape: {original_output.shape}")
    log_message(f"Expanded output shape: {expanded_output.shape}")
    log_message(f"Outputs are different: {not torch.allclose(original_output, expanded_output, atol=1e-6)}")
    
    # Check trainable parameters
    trainable_params = sum(p.numel() for p in expanded_ffn.parameters() if p.requires_grad)
    log_message(f"Trainable parameters in ExpandedFFN: {trainable_params:,}")
    
    return True

def test_small_training():
    """Test small training run"""
    log_message("Testing small training run...")
    
    set_seed(42)
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create small dataset
    train_data = [
        {
            'func_name': 'hello_world',
            'docstring': 'Print hello world',
            'code': 'print("Hello, World!")'
        },
        {
            'func_name': 'add_numbers',
            'docstring': 'Add two numbers',
            'code': 'return a + b'
        }
    ] * 5  # Repeat for more samples
    
    # Initialize learner
    learner = FFNExpansionContinualLearner(model_name, tokenizer, device, expansion_size=128)
    learner.prepare_model()
    
    # Create expanded model
    expanded_model = expand_model_ffn(learner.base_model, 128)
    
    # Test training for 1 epoch with small batch
    training_time = learner._train_model(expanded_model, train_data, epochs=1, batch_size=2)
    
    log_message(f"Training completed in {training_time:.2f} minutes")
    
    # Test evaluation
    results = learner._evaluate_model(expanded_model, train_data[:2], 2, "python")
    log_message(f"Evaluation results: {results}")
    
    return True

def main():
    """Run all tests"""
    log_message("=== TESTING FFN EXPANSION FIXES ===")
    
    try:
        test_ffn_structure()
        test_expanded_ffn_creation()
        test_small_training()
        log_message("✅ All tests passed!")
    except Exception as e:
        log_message(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 