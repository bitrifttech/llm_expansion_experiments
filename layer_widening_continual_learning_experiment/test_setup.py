import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffn_expansion_continual_learning import (
    ExpandedFFN, expand_model_ffn, FFNExpansionContinualLearner,
    log_message, device
)

# Import ModelAnalyzer
from utils.model_analyzer import ModelAnalyzer, analyze_model

def test_expanded_ffn():
    """Test the ExpandedFFN module"""
    log_message("Testing ExpandedFFN module...")
    
    # Create a simple FFN-like module for testing
    class SimpleDenseReluDense(torch.nn.Module):
        def __init__(self, d_model=512, d_ff=2048):
            super().__init__()
            self.wi_0 = torch.nn.Linear(d_model, d_ff, bias=False)
            self.wi_1 = torch.nn.Linear(d_model, d_ff, bias=False)
            self.wo = torch.nn.Linear(d_ff, d_model, bias=False)
            self.dropout = torch.nn.Dropout(0.1)
            self.act = torch.nn.ReLU()
            
        def forward(self, x):
            hidden_gelu = self.act(self.wi_0(x))
            hidden_linear = self.wi_1(x)
            hidden_states = hidden_gelu * hidden_linear
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.wo(hidden_states)
            return hidden_states
    
    # Create original FFN
    original_ffn = SimpleDenseReluDense()
    
    # Create expanded FFN
    expanded_ffn = ExpandedFFN(original_ffn, expansion_size=256, device="cpu")
    
    # Test forward pass
    batch_size, seq_len, d_model = 2, 10, 512
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    # Original output
    original_output = original_ffn(test_input)
    
    # Expanded output
    expanded_output = expanded_ffn(test_input)
    
    # Check shapes
    assert original_output.shape == expanded_output.shape, f"Shape mismatch: {original_output.shape} vs {expanded_output.shape}"
    
    # Check that outputs are different (expansion should add something)
    assert not torch.allclose(original_output, expanded_output, atol=1e-6), "Expanded output should differ from original"
    
    # Check parameter counts
    original_params = sum(p.numel() for p in original_ffn.parameters())
    expanded_params = sum(p.numel() for p in expanded_ffn.parameters() if p.requires_grad)
    
    log_message(f"Original FFN parameters: {original_params:,}")
    log_message(f"Expansion parameters: {expanded_params:,}")
    log_message(f"Expansion ratio: {expanded_params/original_params:.2f}x")
    
    log_message("✅ ExpandedFFN test passed!")
    return True

def test_model_expansion():
    """Test expanding a full T5 model"""
    log_message("Testing full model expansion...")
    
    # Load a small model for testing
    model_name = "Salesforce/codet5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Analyze original model
    original_analyzer = ModelAnalyzer(model, "Original T5-Small")
    original_analysis = original_analyzer.analyze(detailed=False)
    
    # Expand the model
    expanded_model = expand_model_ffn(model, expansion_size=256)  # Smaller for testing
    
    # Verify that original parameters are frozen
    total_params = sum(p.numel() for p in expanded_model.parameters())
    trainable_params = sum(p.numel() for p in expanded_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    assert frozen_params == original_analysis.total_parameters, f"Frozen params {frozen_params} should equal original {original_analysis.total_parameters}"
    
    # Test forward pass
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_input = "def hello_world():"
    test_target = "print('Hello, World!')"
    
    input_encoding = tokenizer(test_input, return_tensors="pt").to(device)
    target_encoding = tokenizer(test_target, return_tensors="pt").to(device)
    
    # Test that the model can run forward pass
    with torch.no_grad():
        outputs = expanded_model(
            input_ids=input_encoding.input_ids,
            labels=target_encoding.input_ids
        )
        loss = outputs.loss
        log_message(f"Test forward pass loss: {loss.item():.4f}")
    
    # Test generation
    with torch.no_grad():
        generated = expanded_model.generate(
            input_ids=input_encoding.input_ids,
            max_length=50,
            num_beams=2,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        log_message(f"Test generation: '{generated_text}'")
    
    log_message("✅ Model expansion test passed!")
    return True

def test_learner_initialization():
    """Test the FFNExpansionContinualLearner class"""
    log_message("Testing FFNExpansionContinualLearner initialization...")
    
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize learner
    learner = FFNExpansionContinualLearner(model_name, tokenizer, device, expansion_size=256)
    learner.prepare_model()
    
    # Check that base model is loaded
    assert learner.base_model is not None, "Base model should be loaded"
    
    # Check parameter count
    base_params = sum(p.numel() for p in learner.base_model.parameters())
    log_message(f"Learner base model parameters: {base_params:,}")
    
    log_message("✅ Learner initialization test passed!")
    return True

def test_small_training():
    """Test training on a very small dataset"""
    log_message("Testing small training run...")
    
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create tiny dataset
    tiny_data = [
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
    ]
    
    # Initialize learner
    learner = FFNExpansionContinualLearner(model_name, tokenizer, device, expansion_size=128)
    learner.prepare_model()
    
    # Test training (1 epoch, very small batch)
    try:
        training_time = learner.train_task(tiny_data, "test", epochs=1, batch_size=1)
        log_message(f"Small training completed in {training_time:.2f} minutes")
        
        # Test evaluation
        results = learner.evaluate_task(tiny_data, "test", num_samples=2)
        log_message(f"Small evaluation results: BLEU={results['bleu']:.4f}, Pass Rate={results['pass_rate']:.2%}")
        
        log_message("✅ Small training test passed!")
        return True
        
    except Exception as e:
        log_message(f"❌ Small training test failed: {e}", level="ERROR")
        return False

def run_all_tests():
    """Run all tests"""
    log_message("Starting FFN Expansion setup tests...")
    log_message(f"Using device: {device}")
    
    tests = [
        ("ExpandedFFN Module", test_expanded_ffn),
        ("Model Expansion", test_model_expansion),
        ("Learner Initialization", test_learner_initialization),
        ("Small Training", test_small_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        log_message(f"\n--- Running {test_name} Test ---")
        try:
            if test_func():
                passed += 1
            else:
                log_message(f"❌ {test_name} test failed", level="ERROR")
        except Exception as e:
            log_message(f"❌ {test_name} test failed with exception: {e}", level="ERROR")
    
    log_message(f"\n=== TEST SUMMARY ===")
    log_message(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        log_message("🎉 All tests passed! Ready to run the full experiment.")
        return True
    else:
        log_message("❌ Some tests failed. Please fix issues before running the full experiment.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 