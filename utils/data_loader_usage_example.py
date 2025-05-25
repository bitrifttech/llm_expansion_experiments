#!/usr/bin/env python3
"""
Example usage of the CodeSearchNetDataLoader utility.

This demonstrates how to replace the existing load_and_prepare_data functions
in the experiments with the new unified data loader.
"""

import sys
import os

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import CodeSearchNetDataLoader, load_and_prepare_data

def example_original_style():
    """Example showing drop-in replacement for original load_and_prepare_data"""
    print("=== Original Style Usage (Drop-in Replacement) ===")
    
    # This is exactly how the original experiments called the function
    python_train, python_val, js_train, js_val = load_and_prepare_data()
    
    print(f"Loaded data: Python train={len(python_train)}, val={len(python_val)}")
    print(f"             JavaScript train={len(js_train)}, val={len(js_val)}")
    
    # Access data the same way as before
    print(f"First Python sample: {python_train[0]['func_code_string'][:50]}...")
    print(f"First JS sample: {js_train[0]['func_code_string'][:50]}...")

def example_class_based_usage():
    """Example showing the new class-based approach with more control"""
    print("\n=== Class-Based Usage (More Control) ===")
    
    # Create loader with custom sizes
    loader = CodeSearchNetDataLoader(
        python_train_size=1000,
        python_val_size=200,
        js_train_size=1000,
        js_val_size=200,
        seed=42
    )
    
    # Load in HuggingFace format (same as original)
    python_train, python_val, js_train, js_val = loader.load_data(format_type="huggingface")
    
    print(f"Custom sizes: Python train={len(python_train)}, val={len(python_val)}")
    print(f"              JavaScript train={len(js_train)}, val={len(js_val)}")
    
    # Get statistics
    stats = loader.get_data_stats()
    print(f"Data stats: {stats}")
    
    # Validate consistency
    is_consistent = loader.validate_data_consistency()
    print(f"Data consistency: {'✅ PASS' if is_consistent else '❌ FAIL'}")

def example_different_formats():
    """Example showing different data formats for different experiment needs"""
    print("\n=== Different Format Examples ===")
    
    loader = CodeSearchNetDataLoader(
        python_train_size=100,
        python_val_size=50,
        js_train_size=100,
        js_val_size=50,
        seed=42
    )
    
    # Format 1: HuggingFace datasets (for LoRA vs Full Layer experiment)
    print("1. HuggingFace format (for LoRA vs Full Layer experiment):")
    python_train_hf, python_val_hf, js_train_hf, js_val_hf = loader.load_data(format_type="huggingface")
    print(f"   Type: {type(python_train_hf)}")
    print(f"   Sample keys: {list(python_train_hf[0].keys())}")
    
    # Format 2: Dict format (for FFN/Attention expansion experiments)
    print("\n2. Dict format (for FFN/Attention expansion experiments):")
    python_train_dict, python_val_dict, js_train_dict, js_val_dict = loader.load_data(format_type="dict")
    print(f"   Type: {type(python_train_dict)}")
    print(f"   Sample keys: {list(python_train_dict[0].keys())}")
    print(f"   Sample input: {python_train_dict[0]['input'][:50]}...")
    
    # Format 3: Raw format (for custom processing)
    print("\n3. Raw format (for custom processing):")
    python_train_raw, python_val_raw, js_train_raw, js_val_raw = loader.load_data(format_type="raw")
    print(f"   Type: {type(python_train_raw)}")
    print(f"   Sample keys: {list(python_train_raw[0].keys())}")

def example_experiment_integration():
    """Example showing how to integrate into existing experiment code"""
    print("\n=== Experiment Integration Example ===")
    
    # OLD WAY (in existing experiments):
    # def load_and_prepare_data():
    #     dataset = load_dataset("code_search_net", split="train")
    #     python_data = dataset.filter(lambda x: x["language"] == "python").select(range(20000))
    #     js_data = dataset.filter(lambda x: x["language"] == "javascript").select(range(20000))
    #     python_train = python_data.select(range(15000))
    #     python_val = python_data.select(range(15000, 20000))
    #     js_train = js_data.select(range(15000))
    #     js_val = js_data.select(range(15000, 20000))
    #     return python_train, python_val, js_train, js_val
    
    # NEW WAY (using the utility):
    from utils.data_loader import load_and_prepare_data
    
    # Exact same interface, but now consistent across all experiments
    python_train, python_val, js_train, js_val = load_and_prepare_data(
        python_train_size=15000,
        python_val_size=5000,
        js_train_size=15000,
        js_val_size=5000,
        format_type="huggingface"
    )
    
    print("✅ Successfully loaded data using the new utility!")
    print(f"   Python: {len(python_train)} train, {len(python_val)} val")
    print(f"   JavaScript: {len(js_train)} train, {len(js_val)} val")
    
    # The rest of the experiment code remains exactly the same!

if __name__ == "__main__":
    print("🚀 CodeSearchNetDataLoader Usage Examples")
    print("=" * 50)
    
    try:
        example_original_style()
        example_class_based_usage()
        example_different_formats()
        example_experiment_integration()
        
        print("\n🎉 All examples completed successfully!")
        print("\n📝 Summary:")
        print("   • Use load_and_prepare_data() for drop-in replacement")
        print("   • Use CodeSearchNetDataLoader class for more control")
        print("   • Supports multiple formats: 'huggingface', 'dict', 'raw'")
        print("   • Ensures consistent data splits across all experiments")
        print("   • Provides validation and statistics methods")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 