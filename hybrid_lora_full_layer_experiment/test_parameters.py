#!/usr/bin/env python3
"""Quick test to verify hybrid model parameter counting"""

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from hybrid_experiment import HybridLoRAFullLayerLearner, log_message

def test_parameter_counting():
    """Test that LoRA + Full Layer parameter counting is correct"""
    
    # Setup
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🧪 Testing Hybrid Model Parameter Counting")
    print("=" * 50)
    
    # Initialize learner
    learner = HybridLoRAFullLayerLearner(model_name, tokenizer, device)
    learner.prepare_model()
    
    # Create hybrid model
    print("\n1. Creating hybrid model...")
    hybrid_model = learner.create_hybrid_model(learner.base_model, "test")
    
    # Manual verification
    print("\n2. Manual parameter verification...")
    total_params = sum(p.numel() for p in hybrid_model.parameters())
    trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    lora_params = sum(p.numel() for n, p in hybrid_model.named_parameters() if 'lora' in n and p.requires_grad)
    full_layer_params = sum(p.numel() for n, p in hybrid_model.named_parameters() 
                           if 'encoder.block.6' in n and p.requires_grad and 'lora' not in n)  # Exclude LoRA params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Full layer parameters: {full_layer_params:,}")
    print(f"LoRA + Full layer: {lora_params + full_layer_params:,}")
    
    # Verify math
    print("\n3. Verification:")
    if trainable_params == lora_params + full_layer_params:
        print("✅ Parameter counting is CORRECT!")
        print(f"   Total trainable = LoRA + Full Layer")
        print(f"   {trainable_params:,} = {lora_params:,} + {full_layer_params:,}")
    else:
        print("❌ Parameter counting is INCORRECT!")
        print(f"   Expected: {lora_params + full_layer_params:,}")
        print(f"   Actual: {trainable_params:,}")
        print(f"   Difference: {trainable_params - (lora_params + full_layer_params):,}")
    
    # Check that full layer parameters are actually trainable
    full_layer_trainable_count = 0
    for name, param in hybrid_model.named_parameters():
        if 'encoder.block.6' in name and param.requires_grad:
            full_layer_trainable_count += 1
    
    print(f"\n4. Full layer parameter status:")
    print(f"   Trainable full layer parameters: {full_layer_trainable_count}")
    
    if full_layer_trainable_count > 0:
        print("✅ Full layer parameters are trainable!")
    else:
        print("❌ Full layer parameters are frozen!")

if __name__ == "__main__":
    test_parameter_counting() 