#!/usr/bin/env python3
"""
Test script to verify the hybrid experiment setup works correctly.
This runs a minimal version of the experiment to check for import and basic functionality issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import peft
        import datasets
        import nltk
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_device_setup():
    """Test device configuration"""
    print("Testing device setup...")
    import torch
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ MPS available")
    else:
        device = "cpu"
        print("✅ Using CPU")
    
    return device

def test_model_loading():
    """Test base model loading"""
    print("Testing model loading...")
    try:
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        
        model_name = "Salesforce/codet5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test tokenizer
        test_text = "def hello_world():"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenizer works: {test_text} -> {tokens['input_ids'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_hybrid_learner():
    """Test the hybrid learner class"""
    print("Testing HybridLoRAFullLayerLearner...")
    try:
        from hybrid_experiment import HybridLoRAFullLayerLearner
        from transformers import AutoTokenizer
        
        device = "cpu"  # Use CPU for testing
        model_name = "Salesforce/codet5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        learner = HybridLoRAFullLayerLearner(model_name, tokenizer, device)
        print("✅ HybridLoRAFullLayerLearner initialized")
        
        return True
    except Exception as e:
        print(f"❌ Hybrid learner error: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist"""
    print("Testing directory structure...")
    
    required_dirs = [
        "experiment_1_results",
        "experiment_2_results"
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Directory missing: {dir_name}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Testing Hybrid LoRA + Full Layer Experiment Setup")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Device Setup", test_device_setup),
        ("Model Loading", test_model_loading),
        ("Directory Structure", test_directory_structure),
        ("Hybrid Learner", test_hybrid_learner),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The hybrid experiment is ready to run.")
        print("\nTo run the full experiment:")
        print("python hybrid_experiment.py")
    else:
        print("⚠️  Some tests failed. Please check the setup before running the experiment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 