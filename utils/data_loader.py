"""
Data Loader Utility

Provides consistent data loading and preparation for continual learning experiments.
Ensures all experiments use the same data splits and formats for fair comparison.
"""

import os
import sys
import random
from typing import Dict, List, Tuple, Optional, Union
from datasets import load_dataset
import time

def log_message(message, level="INFO"):
    """Simple logging function"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

class CodeSearchNetDataLoader:
    """
    Unified data loader for CodeSearchNet dataset used across all continual learning experiments.
    
    Provides consistent data splits and formats to ensure fair comparison between different
    continual learning approaches (LoRA, Full Layer, FFN Expansion, etc.).
    """
    
    def __init__(self, 
                 python_train_size: int = 15000,
                 python_val_size: int = 5000,
                 js_train_size: int = 15000,
                 js_val_size: int = 5000,
                 seed: int = 42):
        """
        Initialize the data loader with consistent splits.
        
        Args:
            python_train_size: Number of Python training samples
            python_val_size: Number of Python validation samples  
            js_train_size: Number of JavaScript training samples
            js_val_size: Number of JavaScript validation samples
            seed: Random seed for reproducible splits
        """
        self.python_train_size = python_train_size
        self.python_val_size = python_val_size
        self.js_train_size = js_train_size
        self.js_val_size = js_val_size
        self.seed = seed
        
        # Data storage
        self.python_train = None
        self.python_val = None
        self.js_train = None
        self.js_val = None
        
        # Track if data is loaded
        self._data_loaded = False
        
    def load_data(self, format_type: str = "huggingface") -> Tuple:
        """
        Load and prepare CodeSearchNet data with consistent splits.
        
        Args:
            format_type: Data format to return
                - "huggingface": Returns HuggingFace Dataset objects (default)
                - "dict": Returns list of dictionaries with 'input'/'target' keys
                - "raw": Returns list of dictionaries with original CodeSearchNet fields
                
        Returns:
            Tuple of (python_train, python_val, js_train, js_val) in specified format
        """
        if self._data_loaded and format_type == "huggingface":
            return self.python_train, self.python_val, self.js_train, self.js_val
            
        log_message("Loading CodeSearchNet dataset...")
        
        try:
            # Load the full dataset
            dataset = load_dataset("code_search_net", split="train")
            
            # Filter by language and select required samples
            total_python_needed = self.python_train_size + self.python_val_size
            total_js_needed = self.js_train_size + self.js_val_size
            
            python_data = dataset.filter(lambda x: x["language"] == "python").select(range(total_python_needed))
            js_data = dataset.filter(lambda x: x["language"] == "javascript").select(range(total_js_needed))
            
            # Create consistent splits
            self.python_train = python_data.select(range(self.python_train_size))
            self.python_val = python_data.select(range(self.python_train_size, total_python_needed))
            self.js_train = js_data.select(range(self.js_train_size))
            self.js_val = js_data.select(range(self.js_train_size, total_js_needed))
            
            log_message(f"Dataset loaded: Python train={len(self.python_train)}, val={len(self.python_val)}")
            log_message(f"                JavaScript train={len(self.js_train)}, val={len(self.js_val)}")
            
            self._data_loaded = True
            
            # Convert to requested format
            if format_type == "huggingface":
                return self.python_train, self.python_val, self.js_train, self.js_val
            elif format_type == "dict":
                return self._convert_to_dict_format()
            elif format_type == "raw":
                return self._convert_to_raw_format()
            else:
                raise ValueError(f"Unknown format_type: {format_type}")
                
        except Exception as e:
            log_message(f"Dataset loading error: {e}", level="ERROR")
            raise
    
    def _convert_to_dict_format(self) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Convert to dictionary format with 'input'/'target' keys for training"""
        def convert_dataset(dataset, language):
            converted = []
            for item in dataset:
                if item['func_name'] and item['func_documentation_string'] and item['func_code_string']:
                    # Use docstring as input, code as target (common pattern)
                    converted.append({
                        'input': f"Generate {language} code: {item['func_documentation_string']}",
                        'target': item['func_code_string']
                    })
            return converted
        
        python_train_dict = convert_dataset(self.python_train, "Python")
        python_val_dict = convert_dataset(self.python_val, "Python") 
        js_train_dict = convert_dataset(self.js_train, "JavaScript")
        js_val_dict = convert_dataset(self.js_val, "JavaScript")
        
        return python_train_dict, python_val_dict, js_train_dict, js_val_dict
    
    def _convert_to_raw_format(self) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Convert to raw format preserving original CodeSearchNet fields"""
        def convert_dataset(dataset):
            converted = []
            for item in dataset:
                if item['func_name'] and item['func_documentation_string'] and item['func_code_string']:
                    converted.append({
                        'func_name': item['func_name'],
                        'docstring': item['func_documentation_string'], 
                        'code': item['func_code_string'],
                        'language': item['language']
                    })
            return converted
        
        python_train_raw = convert_dataset(self.python_train)
        python_val_raw = convert_dataset(self.python_val)
        js_train_raw = convert_dataset(self.js_train)
        js_val_raw = convert_dataset(self.js_val)
        
        return python_train_raw, python_val_raw, js_train_raw, js_val_raw
    
    def get_data_stats(self) -> Dict[str, int]:
        """Get statistics about the loaded data"""
        if not self._data_loaded:
            raise RuntimeError("Data not loaded yet. Call load_data() first.")
            
        return {
            'python_train_size': len(self.python_train),
            'python_val_size': len(self.python_val),
            'js_train_size': len(self.js_train),
            'js_val_size': len(self.js_val),
            'total_samples': len(self.python_train) + len(self.python_val) + len(self.js_train) + len(self.js_val)
        }
    
    def validate_data_consistency(self) -> bool:
        """Validate that data splits are consistent and non-overlapping"""
        if not self._data_loaded:
            return False
            
        # Check sizes match expectations
        expected_sizes = {
            'python_train': self.python_train_size,
            'python_val': self.python_val_size, 
            'js_train': self.js_train_size,
            'js_val': self.js_val_size
        }
        
        actual_sizes = {
            'python_train': len(self.python_train),
            'python_val': len(self.python_val),
            'js_train': len(self.js_train), 
            'js_val': len(self.js_val)
        }
        
        for split_name, expected in expected_sizes.items():
            actual = actual_sizes[split_name]
            if actual != expected:
                log_message(f"Size mismatch for {split_name}: expected {expected}, got {actual}", level="WARNING")
                return False
        
        log_message("Data consistency validation passed")
        return True

# Convenience function for backward compatibility
def load_and_prepare_data(python_train_size: int = 15000,
                         python_val_size: int = 5000,
                         js_train_size: int = 15000,
                         js_val_size: int = 5000,
                         format_type: str = "huggingface",
                         seed: int = 42):
    """
    Convenience function that mimics the original load_and_prepare_data functions.
    
    Returns the same format as the original functions for easy drop-in replacement.
    """
    loader = CodeSearchNetDataLoader(
        python_train_size=python_train_size,
        python_val_size=python_val_size,
        js_train_size=js_train_size,
        js_val_size=js_val_size,
        seed=seed
    )
    
    return loader.load_data(format_type=format_type) 