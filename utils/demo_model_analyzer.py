#!/usr/bin/env python3
"""
Demo script for ModelAnalyzer utility

This script demonstrates the capabilities of the ModelAnalyzer by:
1. Analyzing a base T5 model
2. Creating a simple modification (freezing some parameters)
3. Showing the comparison functionality
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_analyzer import ModelAnalyzer, analyze_model

def demo_basic_analysis():
    """Demonstrate basic model analysis"""
    print("🚀 DEMO: Basic Model Analysis")
    print("="*60)
    
    # Load a small model
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")
    
    # Analyze the model
    analysis = analyze_model(model, "CodeT5-Small Demo", detailed=True)
    
    return model, analysis

def demo_parameter_freezing(model):
    """Demonstrate analysis after freezing parameters"""
    print("\n🧊 DEMO: Parameter Freezing Analysis")
    print("="*60)
    
    # Create a copy and freeze embedding layers
    modified_model = model
    
    # Freeze shared embeddings and language model head
    for param in modified_model.shared.parameters():
        param.requires_grad = False
    for param in modified_model.lm_head.parameters():
        param.requires_grad = False
    
    # Analyze the modified model
    modified_analyzer = ModelAnalyzer(modified_model, "CodeT5-Small (Embeddings Frozen)")
    modified_analysis = modified_analyzer.analyze(detailed=False)
    
    return modified_model, modified_analysis

def demo_model_comparison(original_model, modified_model):
    """Demonstrate model comparison functionality"""
    print("\n🔄 DEMO: Model Comparison")
    print("="*60)
    
    # Create analyzers
    original_analyzer = ModelAnalyzer(original_model, "Original Model")
    modified_analyzer = ModelAnalyzer(modified_model, "Modified Model")
    
    # Compare models
    comparison = original_analyzer.compare_with(modified_analyzer, "Parameter Freezing Demo")
    
    return comparison

def demo_custom_layer():
    """Demonstrate custom layer detection"""
    print("\n🔧 DEMO: Custom Layer Detection")
    print("="*60)
    
    # Create a simple model with custom layer
    class CustomExpansionLayer(nn.Module):
        def __init__(self, input_dim, expansion_dim):
            super().__init__()
            self.expansion = nn.Linear(input_dim, expansion_dim)
            self.projection = nn.Linear(expansion_dim, input_dim)
            
        def forward(self, x):
            return self.projection(torch.relu(self.expansion(x)))
    
    # Create a simple model
    class SimpleModelWithCustomLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 512)
            self.custom_layer = CustomExpansionLayer(512, 256)
            self.output = nn.Linear(512, 1000)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.custom_layer(x)
            return self.output(x)
    
    custom_model = SimpleModelWithCustomLayer()
    
    # Analyze the custom model
    analysis = analyze_model(custom_model, "Model with Custom Layer", detailed=True)
    
    return custom_model, analysis

def main():
    """Run all demos"""
    print("🎯 ModelAnalyzer Demonstration")
    print("This demo showcases the capabilities of the ModelAnalyzer utility")
    print("="*80)
    
    # Demo 1: Basic analysis
    original_model, original_analysis = demo_basic_analysis()
    
    # Demo 2: Parameter freezing
    modified_model, modified_analysis = demo_parameter_freezing(original_model)
    
    # Demo 3: Model comparison
    comparison = demo_model_comparison(original_model, modified_model)
    
    # Demo 4: Custom layer detection
    custom_model, custom_analysis = demo_custom_layer()
    
    # Summary
    print("\n📊 DEMO SUMMARY")
    print("="*60)
    print("✅ Basic model analysis - Shows comprehensive model information")
    print("✅ Parameter freezing - Tracks trainable vs frozen parameters")
    print("✅ Model comparison - Before/after analysis with differences")
    print("✅ Custom layer detection - Identifies non-standard components")
    print("\n🎉 ModelAnalyzer demo completed successfully!")
    print("\nThe ModelAnalyzer provides consistent, detailed analysis across all")
    print("continual learning experiments, making it easy to understand model")
    print("modifications and compare different approaches.")

if __name__ == "__main__":
    main() 