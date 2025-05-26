"""
Model Extensions Utility

Provides reusable classes for extending transformer models with:
1. LoRA (Low-Rank Adaptation) adapters
2. Additional transformer layers
3. Hybrid combinations of both

These classes can be used across different experiments for consistent
model extension patterns.
"""

import os
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, Dict, Any, List
from transformers import T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from dataclasses import dataclass
from .experiment_logger import log_message


@dataclass
class ExtensionConfig:
    """Configuration for model extensions"""
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Transformer layer configuration
    layer_position: str = "encoder"  # "encoder", "decoder", or "both"
    layer_initialization_scale: float = 0.01
    
    # General configuration
    freeze_base: bool = True
    save_path: Optional[str] = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]


class LoRAExtension:
    """
    Reusable class for adding LoRA adapters to transformer models.
    
    Handles adapter creation, training, saving, loading, and task switching.
    """
    
    def __init__(self, base_model: nn.Module, config: ExtensionConfig, device_manager=None):
        self.base_model = base_model
        self.config = config
        self.device_manager = device_manager
        self.device = next(base_model.parameters()).device
        self.adapters = {}  # Store adapter paths
        self.current_adapter = None
        
        if config.freeze_base:
            self._freeze_base_model()
    
    def _freeze_base_model(self):
        """Freeze all base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        frozen_params = sum(1 for p in self.base_model.parameters() if not p.requires_grad)
        # Always log now
            log_message(f"Froze {frozen_params} base model parameters")
    
    def create_adapter(self, task_name: str) -> nn.Module:
        """Create a new LoRA adapter for a specific task"""
        # Start with clean base model
        model = deepcopy(self.base_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            task_type="SEQ_2_SEQ_LM",
            lora_dropout=self.config.lora_dropout
        )
        
        # Apply LoRA to model
        lora_model = get_peft_model(model, lora_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in lora_model.parameters())
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for n, p in lora_model.named_parameters() if 'lora' in n and p.requires_grad)
        
        # Always log now
            log_message(
                f"Created LoRA adapter for {task_name}: "
                f"{lora_params:,} LoRA params / {trainable_params:,} trainable / {total_params:,} total"
            )
        
        return lora_model
    
    def save_adapter(self, model: nn.Module, task_name: str, save_path: Optional[str] = None) -> str:
        """Save LoRA adapter to disk"""
        if save_path is None:
            save_path = self.config.save_path or f"adapters/{task_name}"
        
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        self.adapters[task_name] = save_path
        
        # Always log now
            log_message(f"LoRA adapter for {task_name} saved to {save_path}")
        
        return save_path
    
    def load_adapter(self, task_name: str) -> nn.Module:
        """Load LoRA adapter for a specific task"""
        if task_name not in self.adapters:
            raise ValueError(f"No adapter found for task {task_name}")
        
        # Load base model and apply specific adapter
        model = deepcopy(self.base_model)
        adapter_model = PeftModel.from_pretrained(
            model, 
            self.adapters[task_name]
        ).to(self.device)
        
        self.current_adapter = task_name
        
        # Always log now
            log_message(f"Loaded LoRA adapter for {task_name}")
        
        return adapter_model
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get information about available adapters"""
        return {
            'available_adapters': list(self.adapters.keys()),
            'current_adapter': self.current_adapter,
            'config': {
                'lora_r': self.config.lora_r,
                'lora_alpha': self.config.lora_alpha,
                'lora_dropout': self.config.lora_dropout,
                'target_modules': self.config.lora_target_modules
            }
        }


class TransformerLayerExtension:
    """
    Reusable class for adding new transformer layers to models.
    
    Handles layer creation, training, saving, loading, and task switching.
    """
    
    def __init__(self, base_model: nn.Module, config: ExtensionConfig, device_manager=None):
        self.base_model = base_model
        self.config = config
        self.device_manager = device_manager
        self.device = next(base_model.parameters()).device
        self.checkpoints = {}  # Store checkpoint paths
        self.current_checkpoint = None
        
        if config.freeze_base:
            self._freeze_base_model()
    
    def _freeze_base_model(self):
        """Freeze all base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        frozen_params = sum(1 for p in self.base_model.parameters() if not p.requires_grad)
        # Always log now
            log_message(f"Froze {frozen_params} base model parameters")
    
    def create_extended_model(self, task_name: str) -> nn.Module:
        """Create a model with additional transformer layer(s)"""
        # Start with clean base model
        model = deepcopy(self.base_model)
        
        # Add new transformer layer
        extended_model = self._add_transformer_layer(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in extended_model.parameters())
        trainable_params = sum(p.numel() for p in extended_model.parameters() if p.requires_grad)
        
        # Always log now
            log_message(
                f"Created extended model for {task_name}: "
                f"{trainable_params:,} trainable / {total_params:,} total parameters "
                f"({100*trainable_params/total_params:.2f}%)"
            )
        
        return extended_model
    
    def _add_transformer_layer(self, model: nn.Module) -> nn.Module:
        """Add a new trainable transformer layer to the model"""
        try:
            original_config = model.config
            new_config = deepcopy(original_config)
            new_config.num_layers = original_config.num_layers + 1
            
            # Create new model with extended architecture
            new_model = T5ForConditionalGeneration(new_config).to(model.device)
            
            # Copy weights from original model
            with torch.no_grad():
                # Copy encoder layers
                for i in range(original_config.num_layers):
                    new_model.encoder.block[i].load_state_dict(model.encoder.block[i].state_dict())
                
                # Copy decoder layers
                for i in range(original_config.num_decoder_layers):
                    new_model.decoder.block[i].load_state_dict(model.decoder.block[i].state_dict())
                
                # Copy other components
                new_model.shared.load_state_dict(model.shared.state_dict())
                new_model.encoder.final_layer_norm.load_state_dict(model.encoder.final_layer_norm.state_dict())
                new_model.decoder.final_layer_norm.load_state_dict(model.decoder.final_layer_norm.state_dict())
                new_model.lm_head.load_state_dict(model.lm_head.state_dict())
            
            # Freeze all copied parameters
            for param in new_model.parameters():
                param.requires_grad = False
            
            # Make the new layer(s) trainable based on configuration
            if self.config.layer_position in ["encoder", "both"]:
                new_layer_idx = original_config.num_layers
                for param in new_model.encoder.block[new_layer_idx].parameters():
                    param.requires_grad = True
                    param.data = param.data * self.config.layer_initialization_scale
            
            # Note: For T5, decoder layers are typically not extended in the same way
            # as encoder layers, but this could be added if needed
            
            return new_model
            
        except Exception as e:
            # Always log now
                log_message(f"Error creating extended model: {e}", level="ERROR")
            raise
    
    def save_checkpoint(self, model: nn.Module, task_name: str, save_path: Optional[str] = None) -> str:
        """Save model checkpoint to disk"""
        if save_path is None:
            save_path = self.config.save_path or f"checkpoints/{task_name}"
        
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        self.checkpoints[task_name] = save_path
        
        # Always log now
            log_message(f"Model checkpoint for {task_name} saved to {save_path}")
        
        return save_path
    
    def load_checkpoint(self, task_name: str) -> nn.Module:
        """Load model checkpoint for a specific task"""
        if task_name not in self.checkpoints:
            raise ValueError(f"No checkpoint found for task {task_name}")
        
        model = T5ForConditionalGeneration.from_pretrained(
            self.checkpoints[task_name]
        ).to(self.device)
        
        self.current_checkpoint = task_name
        
        # Always log now
            log_message(f"Loaded checkpoint for {task_name}")
        
        return model
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints"""
        return {
            'available_checkpoints': list(self.checkpoints.keys()),
            'current_checkpoint': self.current_checkpoint,
            'config': {
                'layer_position': self.config.layer_position,
                'layer_initialization_scale': self.config.layer_initialization_scale,
                'freeze_base': self.config.freeze_base
            }
        }


class HybridExtension:
    """
    Reusable class for combining LoRA adapters with additional transformer layers.
    
    Provides the functionality used in the hybrid experiment.
    """
    
    def __init__(self, base_model: nn.Module, config: ExtensionConfig, device_manager=None):
        self.base_model = base_model
        self.config = config
        self.device_manager = device_manager
        self.device = next(base_model.parameters()).device
        self.lora_extension = LoRAExtension(base_model, config, device_manager)
        self.layer_extension = TransformerLayerExtension(base_model, config, device_manager)
        self.hybrid_models = {}
        
    def create_hybrid_model(self, task_name: str, use_shared_layer: bool = False, 
                          shared_layer_model: Optional[nn.Module] = None) -> nn.Module:
        """Create a model with both LoRA adapter and additional transformer layer"""
        
        if use_shared_layer and shared_layer_model is not None:
            # Use the shared layer model as starting point
            model_with_layer = deepcopy(shared_layer_model)
            # Always log now
                log_message(f"Using shared layer for {task_name}")
        else:
            # Create new transformer layer
            model_with_layer = self.layer_extension._add_transformer_layer(self.base_model)
            # Always log now
                log_message(f"Created new layer for {task_name}")
        
        # Get the new layer index for later reference
        original_config = self.base_model.config
        new_layer_idx = original_config.num_layers
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            task_type="SEQ_2_SEQ_LM",
            lora_dropout=self.config.lora_dropout
        )
        
        # Apply LoRA to the model with additional layer
        hybrid_model = get_peft_model(model_with_layer, lora_config)
        
        # CRITICAL: Re-enable training for the transformer layer parameters that LoRA froze
        re_enabled_count = 0
        re_enabled_params = 0
        
        for name, param in hybrid_model.named_parameters():
            # Check if this parameter belongs to our new layer
            if f'encoder.block.{new_layer_idx}' in name and not param.requires_grad:
                param.requires_grad = True
                re_enabled_count += 1
                re_enabled_params += param.numel()
        
        # Count final parameters
        total_trainable = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for n, p in hybrid_model.named_parameters() if 'lora' in n and p.requires_grad)
        layer_params = sum(p.numel() for n, p in hybrid_model.named_parameters() 
                          if f'encoder.block.{new_layer_idx}' in n and p.requires_grad and 'lora' not in n)
        
        # Always log now
            log_message(
                f"Hybrid model for {task_name}: "
                f"LoRA={lora_params:,}, Layer={layer_params:,}, Total={total_trainable:,}"
            )
            if re_enabled_count > 0:
                log_message(
                    f"Re-enabled {re_enabled_count} layer parameters ({re_enabled_params:,} params)"
                )
        
        return hybrid_model
    
    def save_hybrid_model(self, model: nn.Module, task_name: str, save_path: Optional[str] = None) -> str:
        """Save hybrid model (LoRA + layer) to disk"""
        if save_path is None:
            save_path = self.config.save_path or f"hybrid/{task_name}"
        
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        self.hybrid_models[task_name] = save_path
        
        # Always log now
            log_message(f"Hybrid model for {task_name} saved to {save_path}")
        
        return save_path
    
    def load_hybrid_model(self, task_name: str, base_with_layer: Optional[nn.Module] = None) -> nn.Module:
        """Load hybrid model for a specific task"""
        if task_name not in self.hybrid_models:
            raise ValueError(f"No hybrid model found for task {task_name}")
        
        if base_with_layer is not None:
            # Load LoRA adapter onto provided base model with layer
            hybrid_model = PeftModel.from_pretrained(
                base_with_layer,
                self.hybrid_models[task_name]
            ).to(self.device)
        else:
            # Load complete hybrid model (this might not work for all cases)
            # This is a simplified approach - in practice, you might need to
            # reconstruct the base model with layer first
            raise NotImplementedError("Loading hybrid model without base_with_layer not implemented")
        
        # Always log now
            log_message(f"Loaded hybrid model for {task_name}")
        
        return hybrid_model


# Convenience functions for backward compatibility
def create_lora_adapter(base_model: nn.Module, task_name: str, config: Optional[ExtensionConfig] = None, 
                       device_manager=None) -> nn.Module:
    """Convenience function to create a LoRA adapter"""
    if config is None:
        config = ExtensionConfig()
    
    lora_ext = LoRAExtension(base_model, config, device_manager)
    return lora_ext.create_adapter(task_name)


def create_extended_model(base_model: nn.Module, task_name: str, config: Optional[ExtensionConfig] = None,
                         device_manager=None) -> nn.Module:
    """Convenience function to create a model with additional transformer layer"""
    if config is None:
        config = ExtensionConfig()
    
    layer_ext = TransformerLayerExtension(base_model, config, device_manager)
    return layer_ext.create_extended_model(task_name)


def create_hybrid_model(base_model: nn.Module, task_name: str, config: Optional[ExtensionConfig] = None,
                       device_manager=None, use_shared_layer: bool = False, 
                       shared_layer_model: Optional[nn.Module] = None) -> nn.Module:
    """Convenience function to create a hybrid model (LoRA + layer)"""
    if config is None:
        config = ExtensionConfig()
    
    hybrid_ext = HybridExtension(base_model, config, device_manager)
    return hybrid_ext.create_hybrid_model(task_name, use_shared_layer, shared_layer_model) 