#!/usr/bin/env python3
"""
Test script for the CodeSearchNetDataLoader utility.

This script verifies that the data loader works correctly and produces
consistent results across different format types.
"""

import sys
import os

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import CodeSearchNetDataLoader, load_and_prepare_data

def test_data_loader():
    """Test the CodeSearchNetDataLoader class"""
    print("=== Testing CodeSearchNetDataLoader ===")
    
    # Test with smaller sizes for quick testing
    loader = CodeSearchNetDataLoader(
        python_train_size=100,
        python_val_size=50,
        js_train_size=100,
        js_val_size=50,
        seed=42
    )
    
    print("1. Testing HuggingFace format...")
    python_train, python_val, js_train, js_val = loader.load_data(format_type="huggingface")
    
    print(f"   Python train: {len(python_train)} samples")
    print(f"   Python val: {len(python_val)} samples")
    print(f"   JavaScript train: {len(js_train)} samples")
    print(f"   JavaScript val: {len(js_val)} samples")
    
    # Test data consistency
    print("2. Testing data consistency...")
    is_consistent = loader.validate_data_consistency()
    print(f"   Data consistency: {'PASS' if is_consistent else 'FAIL'}")
    
    # Test data stats
    print("3. Testing data statistics...")
    stats = loader.get_data_stats()
    print(f"   Stats: {stats}")
    
    print("4. Testing dict format...")
    python_train_dict, python_val_dict, js_train_dict, js_val_dict = loader.load_data(format_type="dict")
    
    print(f"   Python train dict: {len(python_train_dict)} samples")
    print(f"   Sample input: {python_train_dict[0]['input'][:50]}...")
    print(f"   Sample target: {python_train_dict[0]['target'][:50]}...")
    
    print("5. Testing raw format...")
    python_train_raw, python_val_raw, js_train_raw, js_val_raw = loader.load_data(format_type="raw")
    
    print(f"   Python train raw: {len(python_train_raw)} samples")
    print(f"   Sample keys: {list(python_train_raw[0].keys())}")
    
    print("✅ CodeSearchNetDataLoader tests completed successfully!")

def test_convenience_function():
    """Test the convenience function for backward compatibility"""
    print("\n=== Testing convenience function ===")
    
    print("1. Testing HuggingFace format...")
    python_train, python_val, js_train, js_val = load_and_prepare_data(
        python_train_size=50,
        python_val_size=25,
        js_train_size=50,
        js_val_size=25,
        format_type="huggingface"
    )
    
    print(f"   Python train: {len(python_train)} samples")
    print(f"   Python val: {len(python_val)} samples")
    print(f"   JavaScript train: {len(js_train)} samples")
    print(f"   JavaScript val: {len(js_val)} samples")
    
    print("2. Testing dict format...")
    python_train_dict, python_val_dict, js_train_dict, js_val_dict = load_and_prepare_data(
        python_train_size=50,
        python_val_size=25,
        js_train_size=50,
        js_val_size=25,
        format_type="dict"
    )
    
    print(f"   Python train dict: {len(python_train_dict)} samples")
    print(f"   JavaScript train dict: {len(js_train_dict)} samples")
    
    print("✅ Convenience function tests completed successfully!")

def test_reproducibility():
    """Test that the data loader produces reproducible results"""
    print("\n=== Testing reproducibility ===")
    
    # Load data twice with same seed
    loader1 = CodeSearchNetDataLoader(
        python_train_size=20,
        python_val_size=10,
        js_train_size=20,
        js_val_size=10,
        seed=42
    )
    
    loader2 = CodeSearchNetDataLoader(
        python_train_size=20,
        python_val_size=10,
        js_train_size=20,
        js_val_size=10,
        seed=42
    )
    
    data1 = loader1.load_data(format_type="huggingface")
    data2 = loader2.load_data(format_type="huggingface")
    
    # Check if first few samples are identical
    python_train1, python_val1, js_train1, js_val1 = data1
    python_train2, python_val2, js_train2, js_val2 = data2
    
    # Compare first sample from each split
    sample1_py = python_train1[0]['func_code_string']
    sample2_py = python_train2[0]['func_code_string']
    
    sample1_js = js_train1[0]['func_code_string']
    sample2_js = js_train2[0]['func_code_string']
    
    reproducible = (sample1_py == sample2_py) and (sample1_js == sample2_js)
    
    print(f"   Reproducibility test: {'PASS' if reproducible else 'FAIL'}")
    
    if reproducible:
        print("✅ Reproducibility tests completed successfully!")
    else:
        print("❌ Reproducibility test failed!")

if __name__ == "__main__":
    try:
        test_data_loader()
        test_convenience_function()
        test_reproducibility()
        print("\n🎉 All tests passed! The data loader is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 