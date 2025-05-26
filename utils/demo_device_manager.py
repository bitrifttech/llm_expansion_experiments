#!/usr/bin/env python3
"""
Demo script for DeviceManager utility.

This script demonstrates how to use the DeviceManager for consistent
device selection and management across all experiments.
"""

import torch
import numpy as np
from device_manager import DeviceManager, get_device, get_device_manager


def demo_basic_usage():
    """Demonstrate basic DeviceManager usage"""
    print("🔧 Basic DeviceManager Usage")
    print("=" * 50)
    
    # Auto-detect best device
    print("\n1. Auto-detect device:")
    device_manager = DeviceManager()
    print(f"   Selected device: {device_manager.device}")
    print(f"   Device info: {device_manager}")
    
    # Force specific device
    print("\n2. Force CPU device:")
    cpu_manager = DeviceManager(preferred_device="cpu", verbose=False)
    print(f"   Selected device: {cpu_manager.device}")
    
    # Convenience function
    print("\n3. Using convenience function:")
    device = get_device(verbose=False)
    print(f"   Device: {device}")


def demo_memory_management():
    """Demonstrate memory management features"""
    print("\n\n💾 Memory Management")
    print("=" * 50)
    
    device_manager = DeviceManager(verbose=False)
    
    print("\n1. Current memory info:")
    memory_info = device_manager.get_memory_info()
    for key, value in memory_info.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n2. Clear cache:")
    device_manager.clear_cache()
    print("   Cache cleared (if applicable)")


def demo_model_optimization():
    """Demonstrate model optimization features"""
    print("\n\n⚡ Model Optimization")
    print("=" * 50)
    
    device_manager = DeviceManager(verbose=False)
    
    # Create a simple model
    model = torch.nn.Linear(100, 10)
    print(f"\n1. Model before optimization: device={next(model.parameters()).device}")
    
    # Optimize for device
    optimized_model = device_manager.optimize_for_device(model)
    print(f"   Model after optimization: device={next(optimized_model.parameters()).device}")
    
    # Get recommended batch size
    base_batch_size = 16
    recommended_batch_size = device_manager.get_recommended_batch_size(base_batch_size)
    print(f"\n2. Batch size recommendation:")
    print(f"   Base batch size: {base_batch_size}")
    print(f"   Recommended batch size: {recommended_batch_size}")
    
    # Get recommended dtype
    dtype = device_manager.torch_dtype
    print(f"\n3. Recommended dtype: {dtype}")


def demo_seed_management():
    """Demonstrate seed management"""
    print("\n\n🎲 Seed Management")
    print("=" * 50)
    
    device_manager = DeviceManager(verbose=False)
    
    print("\n1. Setting seed for reproducibility:")
    device_manager.set_seed(42)
    
    # Generate some random numbers to show consistency
    print("   Random numbers after seed:")
    print(f"   Python random: {np.random.randint(0, 100)}")
    print(f"   Torch random: {torch.randint(0, 100, (1,)).item()}")


def demo_experiment_integration():
    """Demonstrate how to integrate DeviceManager into experiments"""
    print("\n\n🧪 Experiment Integration Example")
    print("=" * 50)
    
    print("\n1. Initialize device manager for experiment:")
    device_manager = DeviceManager()
    
    print("\n2. Set up experiment:")
    device_manager.set_seed(42)
    
    print("\n3. Create and optimize model:")
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )
    
    # Move to device and optimize
    model = device_manager.optimize_for_device(model)
    
    print(f"   Model device: {next(model.parameters()).device}")
    print(f"   Model dtype: {next(model.parameters()).dtype}")
    
    print("\n4. Get training configuration:")
    batch_size = device_manager.get_recommended_batch_size(16)
    print(f"   Recommended batch size: {batch_size}")
    
    print("\n5. Monitor memory during training simulation:")
    # Simulate some memory usage
    dummy_data = torch.randn(batch_size, 512).to(device_manager.device)
    dummy_targets = torch.randint(0, 10, (batch_size,)).to(device_manager.device)
    
    # Forward pass
    outputs = model(dummy_data)
    loss = torch.nn.functional.cross_entropy(outputs, dummy_targets)
    
    # Check memory after forward pass
    memory_info = device_manager.get_memory_info()
    print("   Memory after forward pass:")
    for key, value in memory_info.items():
        if isinstance(value, float) and "gb" in key.lower():
            print(f"     {key}: {value:.2f} GB")
    
    # Clean up
    del dummy_data, dummy_targets, outputs, loss
    device_manager.clear_cache()


def demo_backward_compatibility():
    """Demonstrate backward compatibility functions"""
    print("\n\n🔄 Backward Compatibility")
    print("=" * 50)
    
    print("\n1. Old-style device selection:")
    # This mimics the old pattern used in experiments
    device = get_device()
    print(f"   Device: {device}")
    
    print("\n2. Get device manager instance:")
    manager = get_device_manager(preferred_device="cpu", verbose=False)
    print(f"   Manager: {manager}")


def demo_error_handling():
    """Demonstrate error handling"""
    print("\n\n⚠️  Error Handling")
    print("=" * 50)
    
    print("\n1. Invalid preferred device:")
    # This should fall back gracefully
    manager = DeviceManager(preferred_device="invalid_device", verbose=True)
    print(f"   Final device: {manager.device}")
    
    print("\n2. Memory info on unavailable device:")
    # This should handle errors gracefully
    memory_info = manager.get_memory_info()
    if "error" in memory_info:
        print(f"   Error handled: {memory_info['error']}")
    else:
        print("   Memory info retrieved successfully")


def main():
    """Run all demonstrations"""
    print("🚀 DeviceManager Demonstration")
    print("=" * 60)
    
    try:
        demo_basic_usage()
        demo_memory_management()
        demo_model_optimization()
        demo_seed_management()
        demo_experiment_integration()
        demo_backward_compatibility()
        demo_error_handling()
        
        print("\n\n✅ All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 