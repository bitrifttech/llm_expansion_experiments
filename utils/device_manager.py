import torch
import psutil
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Information about the selected device"""
    device: str
    device_name: str
    total_memory_gb: float
    available_memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    is_available: bool = True


class DeviceManager:
    """
    Centralized device management for all experiments.
    
    Handles device selection (CUDA, MPS, CPU) with proper logging,
    memory information, and consistent configuration across experiments.
    """
    
    def __init__(self, preferred_device: Optional[str] = None, verbose: bool = True):
        """
        Initialize device manager.
        
        Args:
            preferred_device: Force specific device ("cuda", "mps", "cpu")
            verbose: Enable detailed logging
        """
        self.preferred_device = preferred_device
        self.verbose = verbose
        self.device_info = self._detect_device()
        
        if self.verbose:
            self._log_device_info()
    
    def _detect_device(self) -> DeviceInfo:
        """Detect and select the best available device"""
        
        # If user specified a preferred device, try to use it
        if self.preferred_device:
            if self.preferred_device == "cuda" and torch.cuda.is_available():
                return self._get_cuda_info()
            elif self.preferred_device == "mps" and self._is_mps_available():
                return self._get_mps_info()
            elif self.preferred_device == "cpu":
                return self._get_cpu_info()
            else:
                if self.verbose:
                    self._log_message(f"Preferred device '{self.preferred_device}' not available, falling back to auto-detection", "WARNING")
        
        # Auto-detect best device
        if torch.cuda.is_available():
            return self._get_cuda_info()
        elif self._is_mps_available():
            return self._get_mps_info()
        else:
            return self._get_cpu_info()
    
    def _is_mps_available(self) -> bool:
        """Check if MPS is available with proper error handling"""
        try:
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except Exception:
            return False
    
    def _get_cuda_info(self) -> DeviceInfo:
        """Get CUDA device information"""
        device_name = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        total_memory = properties.total_memory / (1024**3)
        
        # Get available memory
        try:
            torch.cuda.empty_cache()  # Clear cache for accurate reading
            available_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        except Exception:
            available_memory = None
        
        # Get compute capability
        compute_capability = f"{properties.major}.{properties.minor}"
        
        return DeviceInfo(
            device="cuda",
            device_name=device_name,
            total_memory_gb=total_memory,
            available_memory_gb=available_memory,
            compute_capability=compute_capability
        )
    
    def _get_mps_info(self) -> DeviceInfo:
        """Get MPS device information"""
        # MPS uses system memory
        system_memory = psutil.virtual_memory().total / (1024**3)
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        return DeviceInfo(
            device="mps",
            device_name="Apple Silicon MPS",
            total_memory_gb=system_memory,
            available_memory_gb=available_memory
        )
    
    def _get_cpu_info(self) -> DeviceInfo:
        """Get CPU device information"""
        system_memory = psutil.virtual_memory().total / (1024**3)
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        return DeviceInfo(
            device="cpu",
            device_name="CPU",
            total_memory_gb=system_memory,
            available_memory_gb=available_memory
        )
    
    def _log_message(self, message: str, level: str = "INFO"):
        """Log message using experiment logger"""
        if self.verbose:
            try:
                from .experiment_logger import log_message
                log_message(message, level)
            except ImportError:
                # Fallback to simple print if experiment_logger not available
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] [{level}] {message}")
    
    def _log_device_info(self):
        """Log detailed device information"""
        info = self.device_info
        
        self._log_message(f"Device selected: {info.device.upper()}")
        
        if info.device == "cuda":
            self._log_message(f"GPU: {info.device_name}")
            self._log_message(f"GPU Memory: {info.total_memory_gb:.2f} GB total")
            if info.available_memory_gb:
                self._log_message(f"GPU Memory Available: {info.available_memory_gb:.2f} GB")
            if info.compute_capability:
                self._log_message(f"Compute Capability: {info.compute_capability}")
        
        elif info.device == "mps":
            self._log_message(f"Using Apple Silicon MPS acceleration")
            self._log_message(f"System Memory: {info.total_memory_gb:.2f} GB total")
            if info.available_memory_gb:
                self._log_message(f"System Memory Available: {info.available_memory_gb:.2f} GB")
        
        else:  # CPU
            self._log_message(f"Using CPU (no GPU acceleration available)")
            self._log_message(f"System Memory: {info.total_memory_gb:.2f} GB total")
            if info.available_memory_gb:
                self._log_message(f"System Memory Available: {info.available_memory_gb:.2f} GB")
    
    @property
    def device(self) -> str:
        """Get the selected device string"""
        return self.device_info.device
    
    @property
    def torch_dtype(self) -> torch.dtype:
        """Get recommended torch dtype for the device"""
        if self.device_info.device == "cuda":
            return torch.float16  # Use half precision on CUDA for memory efficiency
        else:
            return torch.float32  # Use full precision on MPS/CPU
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        if self.device_info.device == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "total_gb": total,
                    "free_gb": total - reserved
                }
            except Exception:
                return {"error": "Could not get CUDA memory info"}
        else:
            # System memory for MPS/CPU
            memory = psutil.virtual_memory()
            return {
                "used_gb": memory.used / (1024**3),
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3),
                "percent_used": memory.percent
            }
    
    def clear_cache(self):
        """Clear device cache if applicable"""
        if self.device_info.device == "cuda":
            torch.cuda.empty_cache()
            if self.verbose:
                self._log_message("CUDA cache cleared")
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility on the selected device"""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if self.device_info.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        if self.verbose:
            self._log_message(f"Random seed set to {seed} for {self.device_info.device}")
    
    def optimize_for_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply device-specific optimizations to a model"""
        model = model.to(self.device_info.device)
        
        if self.device_info.device == "cuda":
            # Enable optimizations for CUDA
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
            if self.verbose:
                self._log_message("Applied CUDA optimizations")
        
        elif self.device_info.device == "mps":
            # MPS-specific optimizations could go here
            if self.verbose:
                self._log_message("Applied MPS optimizations")
        
        return model
    
    def get_recommended_batch_size(self, base_batch_size: int = 16) -> int:
        """Get recommended batch size based on device capabilities"""
        if self.device_info.device == "cuda":
            # Scale based on GPU memory
            if self.device_info.total_memory_gb >= 24:
                return base_batch_size * 2  # High-end GPU
            elif self.device_info.total_memory_gb >= 12:
                return base_batch_size  # Mid-range GPU
            else:
                return max(base_batch_size // 2, 1)  # Low-end GPU
        
        elif self.device_info.device == "mps":
            # MPS typically has good memory but slower than CUDA
            return max(base_batch_size // 2, 1)
        
        else:  # CPU
            # Conservative batch size for CPU
            return max(base_batch_size // 4, 1)
    
    def __str__(self) -> str:
        """String representation of device manager"""
        info = self.device_info
        return f"DeviceManager(device={info.device}, name='{info.device_name}', memory={info.total_memory_gb:.1f}GB)"
    
    def __repr__(self) -> str:
        return self.__str__()


# Convenience functions for backward compatibility
def get_device(preferred_device: Optional[str] = None, verbose: bool = True) -> str:
    """Get the best available device string"""
    manager = DeviceManager(preferred_device, verbose)
    return manager.device


def get_device_manager(preferred_device: Optional[str] = None, verbose: bool = True) -> DeviceManager:
    """Get a configured device manager instance"""
    return DeviceManager(preferred_device, verbose)


def log_device_info(device_manager: Optional[DeviceManager] = None):
    """Log device information (for backward compatibility)"""
    if device_manager is None:
        device_manager = DeviceManager(verbose=True)
    else:
        device_manager._log_device_info() 