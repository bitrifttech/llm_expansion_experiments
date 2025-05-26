"""
Demo script for Model Extensions

This script demonstrates how to use the new extension classes:
1. LoRAExtension - for adding LoRA adapters
2. TransformerLayerExtension - for adding transformer layers  
3. HybridExtension - for combining both LoRA and transformer layers

These classes provide a clean, reusable interface for model extensions
across different experiments.
"""

import os
import sys
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.device_manager import DeviceManager
from utils.model_extensions import LoRAExtension, TransformerLayerExtension, HybridExtension, ExtensionConfig

def demo_lora_extension():
    """Demonstrate LoRAExtension usage"""
    print("\n" + "="*60)
    print("DEMO: LoRAExtension")
    print("="*60)
    
    # Initialize device manager and model
    device_manager = DeviceManager()
    model_name = "google/flan-t5-small"
    
    print(f"Loading model: {model_name}")
    base_model = T5ForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=device_manager.torch_dtype
    ).to(device_manager.device)
    base_model = device_manager.optimize_for_device(base_model)
    
    # Create LoRA extension configuration
    config = ExtensionConfig(
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=["q", "k", "v", "o"],
        freeze_base=True
    )
    
    # Initialize LoRA extension
    lora_extension = LoRAExtension(base_model, config, device_manager)
    
    # Create adapters for different tasks
    print("\nCreating LoRA adapters...")
    python_adapter = lora_extension.create_adapter("python_coding")
    js_adapter = lora_extension.create_adapter("javascript_coding")
    
    # Show parameter counts
    python_trainable = sum(p.numel() for p in python_adapter.parameters() if p.requires_grad)
    js_trainable = sum(p.numel() for p in js_adapter.parameters() if p.requires_grad)
    
    print(f"Python adapter trainable parameters: {python_trainable:,}")
    print(f"JavaScript adapter trainable parameters: {js_trainable:,}")
    
    # Save and load adapters
    print("\nSaving adapters...")
    lora_extension.save_adapter(python_adapter, "python_coding")
    lora_extension.save_adapter(js_adapter, "javascript_coding")
    
    print("Loading saved adapters...")
    loaded_python = lora_extension.load_adapter("python_coding")
    loaded_js = lora_extension.load_adapter("javascript_coding")
    
    print("✅ LoRA adapters created, saved, and loaded successfully!")
    
    return lora_extension

def demo_transformer_layer_extension():
    """Demonstrate TransformerLayerExtension usage"""
    print("\n" + "="*60)
    print("DEMO: TransformerLayerExtension")
    print("="*60)
    
    # Initialize device manager and model
    device_manager = DeviceManager()
    model_name = "google/flan-t5-small"
    
    print(f"Loading model: {model_name}")
    base_model = T5ForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=device_manager.torch_dtype
    ).to(device_manager.device)
    base_model = device_manager.optimize_for_device(base_model)
    
    original_layers = base_model.config.num_layers
    print(f"Original model has {original_layers} encoder layers")
    
    # Create layer extension configuration
    config = ExtensionConfig(
        freeze_base=True,
        layer_position="encoder",
        layer_initialization_scale=0.01
    )
    
    # Initialize layer extension
    layer_extension = TransformerLayerExtension(base_model, config, device_manager)
    
    # Add layers for different tasks
    print("\nAdding transformer layers...")
    python_model = layer_extension.create_extended_model("python_task")
    js_model = layer_extension.create_extended_model("javascript_task")
    
    # Show layer counts and parameters
    python_layers = python_model.config.num_layers
    js_layers = js_model.config.num_layers
    
    python_trainable = sum(p.numel() for p in python_model.parameters() if p.requires_grad)
    js_trainable = sum(p.numel() for p in js_model.parameters() if p.requires_grad)
    
    print(f"Python model: {python_layers} layers, {python_trainable:,} trainable parameters")
    print(f"JavaScript model: {js_layers} layers, {js_trainable:,} trainable parameters")
    
    print("✅ Transformer layers added successfully!")
    
    return layer_extension

def demo_hybrid_extension():
    """Demonstrate HybridExtension usage"""
    print("\n" + "="*60)
    print("DEMO: HybridExtension")
    print("="*60)
    
    # Initialize device manager and model
    device_manager = DeviceManager()
    model_name = "google/flan-t5-small"
    
    print(f"Loading model: {model_name}")
    base_model = T5ForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=device_manager.torch_dtype
    ).to(device_manager.device)
    base_model = device_manager.optimize_for_device(base_model)
    
    # Create hybrid extension configuration
    config = ExtensionConfig(
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=["q", "k", "v", "o"],
        freeze_base=True,
        layer_position="encoder",
        layer_initialization_scale=0.01
    )
    
    # Initialize hybrid extension
    hybrid_extension = HybridExtension(base_model, config, device_manager)
    
    # Create hybrid models (LoRA + Transformer Layer)
    print("\nCreating hybrid models...")
    python_hybrid = hybrid_extension.create_hybrid_model("python_hybrid")
    js_hybrid = hybrid_extension.create_hybrid_model("javascript_hybrid")
    
    # Show parameter counts
    python_trainable = sum(p.numel() for p in python_hybrid.parameters() if p.requires_grad)
    js_trainable = sum(p.numel() for p in js_hybrid.parameters() if p.requires_grad)
    
    print(f"Python hybrid trainable parameters: {python_trainable:,}")
    print(f"JavaScript hybrid trainable parameters: {js_trainable:,}")
    
    # Count LoRA vs layer parameters
    python_lora_params = sum(p.numel() for n, p in python_hybrid.named_parameters() 
                            if 'lora' in n and p.requires_grad)
    python_layer_params = python_trainable - python_lora_params
    
    print(f"  - LoRA parameters: {python_lora_params:,}")
    print(f"  - Layer parameters: {python_layer_params:,}")
    
    print("✅ Hybrid models (LoRA + Transformer Layer) created successfully!")
    
    return hybrid_extension

def demo_comparison():
    """Compare parameter counts across different extension types"""
    print("\n" + "="*60)
    print("COMPARISON: Parameter Counts")
    print("="*60)
    
    device_manager = DeviceManager()
    model_name = "google/flan-t5-small"
    
    base_model = T5ForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=device_manager.torch_dtype
    ).to(device_manager.device)
    
    base_params = sum(p.numel() for p in base_model.parameters())
    print(f"Base model parameters: {base_params:,}")
    
    # LoRA configuration
    lora_config = ExtensionConfig(lora_r=8, lora_alpha=16, lora_target_modules=["q", "k", "v", "o"])
    lora_ext = LoRAExtension(base_model, lora_config, device_manager)
    lora_model = lora_ext.create_adapter("test")
    lora_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    # Layer configuration
    layer_config = ExtensionConfig(freeze_base=True, layer_position="encoder")
    layer_ext = TransformerLayerExtension(base_model, layer_config, device_manager)
    layer_model = layer_ext.create_extended_model("test")
    layer_trainable = sum(p.numel() for p in layer_model.parameters() if p.requires_grad)
    
    # Hybrid configuration
    hybrid_config = ExtensionConfig(
        lora_r=8, lora_alpha=16, lora_target_modules=["q", "k", "v", "o"],
        freeze_base=True, layer_position="encoder"
    )
    hybrid_ext = HybridExtension(base_model, hybrid_config, device_manager)
    hybrid_model = hybrid_ext.create_hybrid_model("test")
    hybrid_trainable = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    
    print(f"\nTrainable parameters by extension type:")
    print(f"  LoRA only:        {lora_trainable:,} ({100*lora_trainable/base_params:.3f}%)")
    print(f"  Layer only:       {layer_trainable:,} ({100*layer_trainable/base_params:.3f}%)")
    print(f"  Hybrid (LoRA+Layer): {hybrid_trainable:,} ({100*hybrid_trainable/base_params:.3f}%)")
    
    print(f"\nEfficiency comparison:")
    print(f"  Hybrid vs LoRA:   {hybrid_trainable/lora_trainable:.1f}x more parameters")
    print(f"  Hybrid vs Layer:  {hybrid_trainable/layer_trainable:.1f}x more parameters")

def main():
    """Run all demos"""
    print("Model Extensions Demo")
    print("This demo shows how to use the new extension classes for model modifications")
    
    try:
        # Run individual demos
        demo_lora_extension()
        demo_transformer_layer_extension()
        demo_hybrid_extension()
        demo_comparison()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe extension classes provide:")
        print("• Clean, reusable interfaces for model modifications")
        print("• Consistent parameter management and device handling")
        print("• Easy adapter/layer saving and loading")
        print("• Flexible configuration options")
        print("• Automatic optimization for different devices")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 