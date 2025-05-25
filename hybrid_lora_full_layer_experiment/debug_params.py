#!/usr/bin/env python3
"""Debug script to examine parameter names and states"""

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, get_peft_model
from hybrid_experiment import add_trainable_transformer_layer, freeze_base_model

def debug_parameter_changes():
    """Debug what happens to parameters when LoRA is applied"""
    
    model_name = "Salesforce/codet5-small"
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🔍 Debugging Parameter Changes with LoRA")
    print("=" * 60)
    
    # Step 1: Load base model
    print("\n1. Loading base model...")
    base_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    freeze_base_model(base_model)
    
    # Step 2: Add transformer layer  
    print("\n2. Adding transformer layer...")
    model_with_layer = add_trainable_transformer_layer(base_model)
    
    # Examine parameters before LoRA
    print("\n3. Parameters BEFORE LoRA:")
    trainable_before = []
    for name, param in model_with_layer.named_parameters():
        if param.requires_grad:
            trainable_before.append((name, param.numel()))
            if 'encoder.block.6' in name:  # New layer
                print(f"   TRAINABLE: {name} ({param.numel():,} params)")
    
    total_trainable_before = sum(p[1] for p in trainable_before)
    print(f"   Total trainable before LoRA: {total_trainable_before:,}")
    
    # Store new layer parameter names
    new_layer_names = {name for name, param in model_with_layer.named_parameters() 
                       if 'encoder.block.6' in name and param.requires_grad}
    print(f"   New layer parameter names: {len(new_layer_names)}")
    for name in sorted(new_layer_names):
        print(f"     - {name}")
    
    # Step 3: Apply LoRA
    print("\n4. Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        task_type="SEQ_2_SEQ_LM",
        lora_dropout=0.1
    )
    
    hybrid_model = get_peft_model(model_with_layer, lora_config)
    
    # Examine parameters after LoRA
    print("\n5. Parameters AFTER LoRA:")
    trainable_after = []
    lora_params = []
    frozen_new_layer = []
    
    for name, param in hybrid_model.named_parameters():
        if param.requires_grad:
            trainable_after.append((name, param.numel()))
            if 'lora' in name:
                lora_params.append((name, param.numel()))
            elif 'encoder.block.6' in name:
                print(f"   TRAINABLE NEW LAYER: {name} ({param.numel():,} params)")
        else:
            if 'encoder.block.6' in name:
                frozen_new_layer.append((name, param.numel()))
                print(f"   FROZEN NEW LAYER: {name} ({param.numel():,} params)")
    
    total_trainable_after = sum(p[1] for p in trainable_after)
    total_lora = sum(p[1] for p in lora_params)
    total_frozen_new_layer = sum(p[1] for p in frozen_new_layer)
    
    print(f"\n6. Summary:")
    print(f"   Trainable before LoRA: {total_trainable_before:,}")
    print(f"   Trainable after LoRA: {total_trainable_after:,}")
    print(f"   LoRA parameters: {total_lora:,}")
    print(f"   Frozen new layer params: {total_frozen_new_layer:,}")
    
    # Step 4: Try to re-enable new layer parameters
    print("\n7. Re-enabling new layer parameters...")
    re_enabled = 0
    re_enabled_params = 0
    
    for name, param in hybrid_model.named_parameters():
        if 'encoder.block.6' in name and not param.requires_grad:
            param.requires_grad = True
            re_enabled += 1
            re_enabled_params += param.numel()
            print(f"   Re-enabled: {name} ({param.numel():,} params)")
    
    print(f"   Re-enabled {re_enabled} parameters ({re_enabled_params:,} total params)")
    
    # Final count
    final_trainable = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    final_lora = sum(p.numel() for n, p in hybrid_model.named_parameters() if 'lora' in n and p.requires_grad)
    final_new_layer = sum(p.numel() for n, p in hybrid_model.named_parameters() 
                         if 'encoder.block.6' in n and p.requires_grad)
    
    print(f"\n8. Final counts:")
    print(f"   Total trainable: {final_trainable:,}")
    print(f"   LoRA: {final_lora:,}")
    print(f"   New layer: {final_new_layer:,}")
    print(f"   Expected total: {final_lora + final_new_layer:,}")
    
    if final_trainable == final_lora + final_new_layer:
        print("   ✅ Math checks out!")
    else:
        print(f"   ❌ Mismatch: {final_trainable - (final_lora + final_new_layer):,}")

if __name__ == "__main__":
    debug_parameter_changes() 