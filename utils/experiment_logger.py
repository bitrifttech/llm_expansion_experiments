"""
Experiment Logger Utility

Provides centralized logging for all experiments with:
1. File logging to experiment-specific directories
2. Console output for real-time monitoring
3. Automatic log file naming with timestamps
4. Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
5. Structured log formatting
6. Thread-safe logging operations

Usage:
    from utils.experiment_logger import get_experiment_logger
    
    logger = get_experiment_logger(__file__)
    logger.info("This is an info message")
    logger.error("This is an error message")
"""

import os
import sys
import logging
import inspect
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import threading


class ExperimentLogger:
    """
    Centralized logger for experiments that logs to both file and console.
    
    Features:
    - Automatic log file creation in experiment-specific 'logs' directory
    - Console output with colored formatting
    - Thread-safe operations
    - Multiple log levels
    - Structured formatting with timestamps
    """
    
    _instances: Dict[str, 'ExperimentLogger'] = {}
    _lock = threading.Lock()
    
    def __init__(self, experiment_file: str, log_level: str = "INFO"):
        """
        Initialize logger for a specific experiment.
        
        Args:
            experiment_file: Path to the experiment Python file (__file__)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.experiment_file = Path(experiment_file)
        self.experiment_name = self.experiment_file.stem
        self.experiment_dir = self.experiment_file.parent
        self.log_level = getattr(logging, log_level.upper())
        
        # Create logs directory
        self.logs_dir = self.experiment_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{self.experiment_name}_{timestamp}.log"
        self.log_filepath = self.logs_dir / self.log_filename
        
        # Setup logger
        self._setup_logger()
        
        # Log initialization
        self.info(f"Experiment logger initialized for {self.experiment_name}")
        self.info(f"Log file: {self.log_filepath}")
    
    def _setup_logger(self):
        """Setup the Python logging configuration"""
        # Create logger
        self.logger = logging.getLogger(f"experiment_{self.experiment_name}")
        self.logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_filepath, mode='a', encoding='utf-8')
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def log(self, level: str, message: str):
        """Log message with specified level"""
        level_upper = level.upper()
        if hasattr(self.logger, level_upper.lower()):
            getattr(self.logger, level_upper.lower())(message)
        else:
            self.logger.info(f"[{level_upper}] {message}")
    
    def section(self, title: str, char: str = "=", width: int = 60):
        """Log a section header"""
        separator = char * width
        self.info(separator)
        self.info(title.center(width))
        self.info(separator)
    
    def subsection(self, title: str, char: str = "-", width: int = 40):
        """Log a subsection header"""
        separator = char * width
        self.info(separator)
        self.info(title)
        self.info(separator)
    
    def progress(self, current: int, total: int, message: str = "Progress"):
        """Log progress information"""
        percentage = (current / total) * 100 if total > 0 else 0
        self.info(f"{message}: {current}/{total} ({percentage:.1f}%)")
    
    def metrics(self, metrics_dict: Dict, title: str = "Metrics"):
        """Log metrics in a structured format"""
        self.info(f"{title}:")
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
    
    def experiment_start(self, description: str = None):
        """Log experiment start with metadata"""
        self.section("EXPERIMENT START")
        self.info(f"Experiment: {self.experiment_name}")
        self.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"Working Directory: {os.getcwd()}")
        self.info(f"Python Version: {sys.version}")
        if description:
            self.info(f"Description: {description}")
        self.section("", char="=")
    
    def experiment_end(self, success: bool = True, summary: str = None):
        """Log experiment end with metadata"""
        self.section("EXPERIMENT END")
        self.info(f"Experiment: {self.experiment_name}")
        self.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"Status: {'SUCCESS' if success else 'FAILED'}")
        if summary:
            self.info(f"Summary: {summary}")
        self.section("", char="=")
    
    def exception(self, exc: Exception, context: str = None):
        """Log exception with context"""
        if context:
            self.error(f"Exception in {context}: {type(exc).__name__}: {exc}")
        else:
            self.error(f"Exception: {type(exc).__name__}: {exc}")
        
        # Log traceback to file only
        import traceback
        file_handler = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                file_handler = handler
                break
        
        if file_handler:
            # Temporarily remove console handler for traceback
            console_handlers = [h for h in self.logger.handlers if isinstance(h, logging.StreamHandler)]
            for handler in console_handlers:
                self.logger.removeHandler(handler)
            
            # Log full traceback to file
            self.logger.error("Full traceback:")
            for line in traceback.format_exc().splitlines():
                self.logger.error(line)
            
            # Re-add console handlers
            for handler in console_handlers:
                self.logger.addHandler(handler)
    
    def get_log_filepath(self) -> Path:
        """Get the path to the current log file"""
        return self.log_filepath
    
    def get_logs_directory(self) -> Path:
        """Get the logs directory path"""
        return self.logs_dir


def get_experiment_logger(experiment_file: str, log_level: str = "INFO") -> ExperimentLogger:
    """
    Get or create an experiment logger instance.
    
    This function ensures only one logger instance per experiment file,
    making it safe to call multiple times from the same experiment.
    
    Args:
        experiment_file: Path to the experiment Python file (use __file__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        ExperimentLogger instance
    
    Example:
        logger = get_experiment_logger(__file__)
        logger.info("Starting experiment")
    """
    experiment_path = Path(experiment_file).resolve()
    experiment_key = str(experiment_path)
    
    with ExperimentLogger._lock:
        if experiment_key not in ExperimentLogger._instances:
            ExperimentLogger._instances[experiment_key] = ExperimentLogger(experiment_file, log_level)
        return ExperimentLogger._instances[experiment_key]


def log_message(message: str, level: str = "INFO", experiment_file: str = None):
    """
    Backward compatibility function for existing log_message calls.
    
    This function automatically detects the calling experiment file
    and routes to the appropriate logger.
    
    Args:
        message: Log message
        level: Log level (INFO, WARNING, ERROR, etc.)
        experiment_file: Optional experiment file path (auto-detected if None)
    """
    if experiment_file is None:
        # Auto-detect calling file
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            experiment_file = caller_frame.f_globals.get('__file__')
            if experiment_file is None:
                # Fallback to a generic logger
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}")
                return
        finally:
            del frame
    
    logger = get_experiment_logger(experiment_file)
    logger.log(level, message)


# Convenience functions for common logging patterns
def log_device_info(device_info, experiment_file: str = None):
    """Log device information in a structured format"""
    if experiment_file is None:
        frame = inspect.currentframe()
        try:
            experiment_file = frame.f_back.f_globals.get('__file__')
        finally:
            del frame
    
    logger = get_experiment_logger(experiment_file)
    logger.subsection("Device Information")
    logger.info(f"Device: {device_info.device.upper()}")
    logger.info(f"Device Name: {device_info.device_name}")
    logger.info(f"Total Memory: {device_info.total_memory_gb:.2f} GB")
    logger.info(f"Available Memory: {device_info.available_memory_gb:.2f} GB")
    if hasattr(device_info, 'compute_capability') and device_info.compute_capability:
        logger.info(f"Compute Capability: {device_info.compute_capability}")


def log_model_info(model, model_name: str = "Model", experiment_file: str = None):
    """Log model information in a structured format"""
    if experiment_file is None:
        frame = inspect.currentframe()
        try:
            experiment_file = frame.f_back.f_globals.get('__file__')
        finally:
            del frame
    
    logger = get_experiment_logger(experiment_file)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.subsection(f"{model_name} Information")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Trainable Percentage: {100 * trainable_params / total_params:.2f}%")


def log_training_progress(epoch: int, total_epochs: int, loss: float, 
                         elapsed_time: float = None, experiment_file: str = None):
    """Log training progress in a structured format"""
    if experiment_file is None:
        frame = inspect.currentframe()
        try:
            experiment_file = frame.f_back.f_globals.get('__file__')
        finally:
            del frame
    
    logger = get_experiment_logger(experiment_file)
    
    message = f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}"
    if elapsed_time is not None:
        message += f" - Time: {elapsed_time:.2f}s"
    
    logger.info(message)


# Example usage and testing
if __name__ == "__main__":
    # Demo the logging functionality
    logger = get_experiment_logger(__file__)
    
    logger.experiment_start("Testing the experiment logger utility")
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    logger.section("Testing Section Headers")
    logger.subsection("Testing Subsection Headers")
    
    logger.progress(50, 100, "Training Progress")
    
    logger.metrics({
        "accuracy": 0.95,
        "loss": 0.05,
        "f1_score": 0.92
    }, "Model Performance")
    
    try:
        raise ValueError("This is a test exception")
    except Exception as e:
        logger.exception(e, "testing exception logging")
    
    logger.experiment_end(True, "Logger testing completed successfully")
    
    print(f"\nLog file created at: {logger.get_log_filepath()}") 