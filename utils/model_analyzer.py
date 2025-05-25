"""
Model Analyzer Utility

Provides comprehensive analysis of transformer models including:
- Overall model statistics
- Layer-by-layer parameter breakdown
- Trainable vs frozen parameter analysis
- Memory usage estimation
- Model architecture visualization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

@dataclass
class LayerInfo:
    """Information about a single layer"""
    name: str
    layer_type: str
    total_params: int
    trainable_params: int
    frozen_params: int
    shape_info: str
    memory_mb: float
    is_custom: bool = False  # For custom layers like ExpandedFFN, LoRA, etc.

@dataclass
class ModelAnalysis:
    """Complete model analysis results"""
    model_name: str
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    total_memory_mb: float
    num_layers: int
    layer_breakdown: List[LayerInfo]
    architecture_summary: Dict[str, Any]
    custom_components: List[str]
    efficiency_metrics: Dict[str, float]

class ModelAnalyzer:
    """Comprehensive model analyzer for transformer models"""
    
    def __init__(self, model: nn.Module, model_name: str = "Unknown"):
        self.model = model
        self.model_name = model_name
        self.device = next(model.parameters()).device if list(model.parameters()) else "cpu"
        
    def analyze(self, detailed: bool = True) -> ModelAnalysis:
        """
        Perform comprehensive model analysis
        
        Args:
            detailed: Whether to include detailed layer-by-layer analysis
            
        Returns:
            ModelAnalysis object with complete analysis results
        """
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING MODEL: {self.model_name}")
        print(f"{'='*60}")
        
        # Basic parameter counts
        total_params, trainable_params, frozen_params = self._count_parameters()
        
        # Memory analysis
        total_memory = self._estimate_memory()
        
        # Layer breakdown
        layer_breakdown = []
        if detailed:
            layer_breakdown = self._analyze_layers()
        
        # Architecture summary
        arch_summary = self._get_architecture_summary()
        
        # Custom components detection
        custom_components = self._detect_custom_components()
        
        # Efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            total_params, trainable_params, total_memory
        )
        
        analysis = ModelAnalysis(
            model_name=self.model_name,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            frozen_parameters=frozen_params,
            total_memory_mb=total_memory,
            num_layers=len(layer_breakdown),
            layer_breakdown=layer_breakdown,
            architecture_summary=arch_summary,
            custom_components=custom_components,
            efficiency_metrics=efficiency_metrics
        )
        
        self._print_analysis(analysis)
        return analysis
    
    def _count_parameters(self) -> Tuple[int, int, int]:
        """Count total, trainable, and frozen parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return total_params, trainable_params, frozen_params
    
    def _estimate_memory(self) -> float:
        """Estimate model memory usage in MB"""
        total_memory = 0
        
        for param in self.model.parameters():
            # Parameter memory
            param_memory = param.numel() * param.element_size()
            total_memory += param_memory
            
            # Gradient memory (if trainable)
            if param.requires_grad:
                total_memory += param_memory
        
        return total_memory / (1024 * 1024)  # Convert to MB
    
    def _analyze_layers(self) -> List[LayerInfo]:
        """Analyze each layer in detail"""
        layer_info = []
        
        def analyze_module(module, name, prefix=""):
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Count parameters for this module
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            # Skip if no parameters
            if total_params == 0:
                return
            
            # Get layer type and shape info
            layer_type = type(module).__name__
            shape_info = self._get_shape_info(module)
            
            # Estimate memory for this layer
            memory_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024 * 1024)
            if any(p.requires_grad for p in module.parameters()):
                memory_mb *= 2  # Account for gradients
            
            # Detect custom components
            is_custom = self._is_custom_component(module)
            
            layer_info.append(LayerInfo(
                name=full_name,
                layer_type=layer_type,
                total_params=total_params,
                trainable_params=trainable_params,
                frozen_params=frozen_params,
                shape_info=shape_info,
                memory_mb=memory_mb,
                is_custom=is_custom
            ))
        
        # Recursively analyze all modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                analyze_module(module, name)
        
        return layer_info
    
    def _get_shape_info(self, module: nn.Module) -> str:
        """Get shape information for a module"""
        if isinstance(module, nn.Linear):
            return f"Linear({module.in_features} → {module.out_features})"
        elif isinstance(module, nn.Embedding):
            return f"Embedding({module.num_embeddings}, {module.embedding_dim})"
        elif isinstance(module, nn.LayerNorm):
            return f"LayerNorm({module.normalized_shape})"
        elif isinstance(module, nn.MultiheadAttention):
            return f"MultiheadAttention(embed_dim={module.embed_dim}, num_heads={module.num_heads})"
        elif hasattr(module, 'weight') and module.weight is not None:
            return f"Weight shape: {tuple(module.weight.shape)}"
        else:
            return "No shape info"
    
    def _is_custom_component(self, module: nn.Module) -> bool:
        """Detect if a module is a custom component (LoRA, ExpandedFFN, etc.)"""
        custom_types = [
            'ExpandedFFN', 'LoraLayer', 'PeftModel', 'LoraLinear',
            'AdaLoraLayer', 'IA3Layer', 'AdaptionPromptLayer'
        ]
        return any(custom_type in type(module).__name__ for custom_type in custom_types)
    
    def _get_architecture_summary(self) -> Dict[str, Any]:
        """Get high-level architecture summary"""
        summary = {
            'model_type': type(self.model).__name__,
            'device': str(self.device),
            'dtype': str(next(self.model.parameters()).dtype) if list(self.model.parameters()) else 'unknown'
        }
        
        # T5-specific analysis
        if hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
            summary['architecture'] = 'Encoder-Decoder'
            if hasattr(self.model.encoder, 'block'):
                summary['encoder_layers'] = len(self.model.encoder.block)
            if hasattr(self.model.decoder, 'block'):
                summary['decoder_layers'] = len(self.model.decoder.block)
            if hasattr(self.model.config, 'd_model'):
                summary['d_model'] = self.model.config.d_model
            if hasattr(self.model.config, 'd_ff'):
                summary['d_ff'] = self.model.config.d_ff
        
        return summary
    
    def _detect_custom_components(self) -> List[str]:
        """Detect custom components in the model"""
        custom_components = []
        
        for name, module in self.model.named_modules():
            if self._is_custom_component(module):
                custom_components.append(f"{name}: {type(module).__name__}")
        
        return custom_components
    
    def _calculate_efficiency_metrics(self, total_params: int, trainable_params: int, 
                                    total_memory: float) -> Dict[str, float]:
        """Calculate efficiency metrics"""
        metrics = {}
        
        if total_params > 0:
            metrics['trainable_percentage'] = (trainable_params / total_params) * 100
            metrics['frozen_percentage'] = ((total_params - trainable_params) / total_params) * 100
            metrics['params_per_mb'] = total_params / total_memory if total_memory > 0 else 0
        
        metrics['memory_per_param_bytes'] = (total_memory * 1024 * 1024) / total_params if total_params > 0 else 0
        
        return metrics
    
    def _print_analysis(self, analysis: ModelAnalysis):
        """Print formatted analysis results"""
        
        # Header
        print(f"\n📊 MODEL OVERVIEW")
        print(f"├── Model: {analysis.model_name}")
        print(f"├── Type: {analysis.architecture_summary.get('model_type', 'Unknown')}")
        print(f"├── Architecture: {analysis.architecture_summary.get('architecture', 'Unknown')}")
        print(f"├── Device: {analysis.architecture_summary.get('device', 'Unknown')}")
        print(f"└── Data Type: {analysis.architecture_summary.get('dtype', 'Unknown')}")
        
        # Parameter Summary
        print(f"\n🔢 PARAMETER SUMMARY")
        print(f"├── Total Parameters: {analysis.total_parameters:,}")
        print(f"├── Trainable Parameters: {analysis.trainable_parameters:,} ({analysis.efficiency_metrics.get('trainable_percentage', 0):.2f}%)")
        print(f"├── Frozen Parameters: {analysis.frozen_parameters:,} ({analysis.efficiency_metrics.get('frozen_percentage', 0):.2f}%)")
        print(f"└── Memory Usage: {analysis.total_memory_mb:.2f} MB")
        
        # Architecture Details
        if 'encoder_layers' in analysis.architecture_summary or 'decoder_layers' in analysis.architecture_summary:
            print(f"\n🏗️  ARCHITECTURE DETAILS")
            if 'encoder_layers' in analysis.architecture_summary:
                print(f"├── Encoder Layers: {analysis.architecture_summary['encoder_layers']}")
            if 'decoder_layers' in analysis.architecture_summary:
                print(f"├── Decoder Layers: {analysis.architecture_summary['decoder_layers']}")
            if 'd_model' in analysis.architecture_summary:
                print(f"├── Model Dimension: {analysis.architecture_summary['d_model']}")
            if 'd_ff' in analysis.architecture_summary:
                print(f"└── FFN Dimension: {analysis.architecture_summary['d_ff']}")
        
        # Custom Components
        if analysis.custom_components:
            print(f"\n🔧 CUSTOM COMPONENTS")
            for i, component in enumerate(analysis.custom_components):
                prefix = "├──" if i < len(analysis.custom_components) - 1 else "└──"
                print(f"{prefix} {component}")
        
        # Efficiency Metrics
        print(f"\n⚡ EFFICIENCY METRICS")
        print(f"├── Parameters per MB: {analysis.efficiency_metrics.get('params_per_mb', 0):,.0f}")
        print(f"├── Memory per Parameter: {analysis.efficiency_metrics.get('memory_per_param_bytes', 0):.2f} bytes")
        print(f"└── Trainable Ratio: {analysis.efficiency_metrics.get('trainable_percentage', 0):.2f}%")
        
        # Layer Breakdown (top trainable layers)
        if analysis.layer_breakdown:
            trainable_layers = [layer for layer in analysis.layer_breakdown if layer.trainable_params > 0]
            if trainable_layers:
                print(f"\n🎯 TOP TRAINABLE LAYERS")
                trainable_layers.sort(key=lambda x: x.trainable_params, reverse=True)
                for i, layer in enumerate(trainable_layers[:5]):  # Top 5
                    prefix = "├──" if i < min(4, len(trainable_layers) - 1) else "└──"
                    custom_marker = " 🔧" if layer.is_custom else ""
                    print(f"{prefix} {layer.name}: {layer.trainable_params:,} params ({layer.layer_type}){custom_marker}")
                
                if len(trainable_layers) > 5:
                    print(f"    ... and {len(trainable_layers) - 5} more trainable layers")
        
        print(f"\n{'='*60}")
    
    def compare_with(self, other_analyzer: 'ModelAnalyzer', comparison_name: str = "Comparison"):
        """Compare this model with another model"""
        print(f"\n{'='*60}")
        print(f"🔄 MODEL COMPARISON: {comparison_name}")
        print(f"{'='*60}")
        
        # Get analyses
        analysis1 = self.analyze(detailed=False)
        analysis2 = other_analyzer.analyze(detailed=False)
        
        # Parameter comparison
        param_diff = analysis2.total_parameters - analysis1.total_parameters
        trainable_diff = analysis2.trainable_parameters - analysis1.trainable_parameters
        memory_diff = analysis2.total_memory_mb - analysis1.total_memory_mb
        
        print(f"\n📈 PARAMETER CHANGES")
        print(f"├── Total Parameters: {analysis1.total_parameters:,} → {analysis2.total_parameters:,} ({param_diff:+,})")
        print(f"├── Trainable Parameters: {analysis1.trainable_parameters:,} → {analysis2.trainable_parameters:,} ({trainable_diff:+,})")
        print(f"├── Memory Usage: {analysis1.total_memory_mb:.2f} MB → {analysis2.total_memory_mb:.2f} MB ({memory_diff:+.2f} MB)")
        
        if analysis1.total_parameters > 0:
            param_increase_pct = (param_diff / analysis1.total_parameters) * 100
            print(f"└── Parameter Increase: {param_increase_pct:+.2f}%")
        
        # Custom components comparison
        new_components = set(analysis2.custom_components) - set(analysis1.custom_components)
        if new_components:
            print(f"\n🆕 NEW CUSTOM COMPONENTS")
            for component in new_components:
                print(f"└── {component}")
        
        print(f"\n{'='*60}")
        
        return {
            'original': analysis1,
            'modified': analysis2,
            'parameter_diff': param_diff,
            'trainable_diff': trainable_diff,
            'memory_diff': memory_diff
        }

def analyze_model(model: nn.Module, model_name: str = "Model", detailed: bool = True) -> ModelAnalysis:
    """
    Convenience function to analyze a model
    
    Args:
        model: PyTorch model to analyze
        model_name: Name for the model
        detailed: Whether to include detailed layer analysis
        
    Returns:
        ModelAnalysis object
    """
    analyzer = ModelAnalyzer(model, model_name)
    return analyzer.analyze(detailed=detailed) 