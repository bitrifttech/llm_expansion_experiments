"""
Test suite for ModelEvaluator

This module contains comprehensive tests for the ModelEvaluator utility class
to ensure it works correctly and handles edge cases properly.
"""

import unittest
import sys
import os
import torch
import numpy as np
from unittest.mock import Mock, patch

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_evaluator import (
    ModelEvaluator, 
    ContinualLearningEvaluator, 
    EvaluationConfig, 
    EvaluationResults,
    evaluate_model_basic,
    evaluate_model_comprehensive
)


class TestEvaluationConfig(unittest.TestCase):
    """Test EvaluationConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EvaluationConfig()
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.num_beams, 3)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.default_num_samples, 100)
        self.assertEqual(config.input_fraction, 0.5)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = EvaluationConfig(
            max_length=256,
            num_beams=2,
            temperature=0.5,
            default_num_samples=50
        )
        self.assertEqual(config.max_length, 256)
        self.assertEqual(config.num_beams, 2)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.default_num_samples, 50)


class TestEvaluationResults(unittest.TestCase):
    """Test EvaluationResults dataclass"""
    
    def test_results_creation(self):
        """Test creating evaluation results"""
        results = EvaluationResults(
            bleu=0.5,
            meteor=0.4,
            pass_rate=0.8,
            edit_distance=0.3,
            ast_similarity=0.6,
            complexity=5.0,
            num_samples=100,
            language="python"
        )
        
        self.assertEqual(results.bleu, 0.5)
        self.assertEqual(results.meteor, 0.4)
        self.assertEqual(results.pass_rate, 0.8)
        self.assertEqual(results.language, "python")
    
    def test_composite_score(self):
        """Test composite score calculation"""
        config = EvaluationConfig()
        results = EvaluationResults(
            bleu=0.5,
            meteor=0.4,
            pass_rate=0.8,
            edit_distance=0.3,
            ast_similarity=0.6,
            complexity=5.0,
            num_samples=100,
            language="python"
        )
        
        score = results.composite_score(config)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)
        self.assertLess(score, 1)
    
    def test_to_dict(self):
        """Test converting results to dictionary"""
        results = EvaluationResults(
            bleu=0.5,
            meteor=0.4,
            pass_rate=0.8,
            edit_distance=0.3,
            ast_similarity=0.6,
            complexity=5.0,
            num_samples=100,
            language="python"
        )
        
        result_dict = results.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['bleu'], 0.5)
        self.assertEqual(result_dict['language'], "python")
        self.assertIn('meteor', result_dict)
        self.assertIn('pass_rate', result_dict)


class TestModelEvaluator(unittest.TestCase):
    """Test ModelEvaluator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.tokenize.return_value = ['def', 'test', '(', ')', ':']
        self.mock_tokenizer.decode.return_value = "def test(): pass"
        
        # Mock model
        self.mock_model = Mock()
        self.mock_model.eval.return_value = None
        self.mock_model.device = "cpu"
        
        # Mock generation output
        mock_output = Mock()
        mock_output.__getitem__.return_value = torch.tensor([1, 2, 3, 4, 5])
        self.mock_model.generate.return_value = mock_output
        
        # Mock tokenizer call
        mock_inputs = Mock()
        mock_inputs.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        self.mock_tokenizer.return_value = mock_inputs
        
        self.evaluator = ModelEvaluator(self.mock_tokenizer)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator.tokenizer)
        self.assertIsNotNone(self.evaluator.config)
        self.assertIsNotNone(self.evaluator.smoothing)
    
    def test_extract_source_code(self):
        """Test source code extraction from different data formats"""
        # Test dictionary format
        data_dict = {"func_code_string": "def test(): pass"}
        result = self.evaluator._extract_source_code(data_dict)
        self.assertEqual(result, "def test(): pass")
        
        # Test string format
        data_str = "def test(): pass"
        result = self.evaluator._extract_source_code(data_str)
        self.assertEqual(result, "def test(): pass")
        
        # Test empty data
        result = self.evaluator._extract_source_code({})
        self.assertIsNone(result)
    
    def test_detect_language(self):
        """Test language detection"""
        # Python code
        python_code = "def test(): import os; print('hello')"
        result = self.evaluator._detect_language(python_code)
        self.assertEqual(result, "python")
        
        # JavaScript code
        js_code = "function test() { var x = 5; console.log(x); }"
        result = self.evaluator._detect_language(js_code)
        self.assertEqual(result, "javascript")
        
        # Unknown code
        unknown_code = "some random text"
        result = self.evaluator._detect_language(unknown_code)
        self.assertEqual(result, "unknown")
    
    def test_prepare_input_target(self):
        """Test input/target preparation"""
        source_code = "def test(): return 42"
        input_text, target_text = self.evaluator._prepare_input_target(source_code)
        
        self.assertLess(len(input_text), len(source_code))
        self.assertEqual(target_text, source_code)
    
    def test_calculate_bleu(self):
        """Test BLEU score calculation"""
        predicted = "def test(): pass"
        target = "def test(): return 42"
        
        score = self.evaluator._calculate_bleu(predicted, target)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_meteor(self):
        """Test METEOR score calculation"""
        predicted = "def test(): pass"
        target = "def test(): return 42"
        
        score = self.evaluator._calculate_meteor(predicted, target)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_edit_distance(self):
        """Test edit distance calculation"""
        predicted = "def test(): pass"
        target = "def test(): return 42"
        
        distance = self.evaluator._calculate_edit_distance(predicted, target)
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, 1.0)
    
    def test_calculate_python_ast_similarity(self):
        """Test Python AST similarity calculation"""
        predicted = "def test(): pass"
        target = "def test(): return 42"
        
        similarity = self.evaluator._calculate_python_ast_similarity(predicted, target)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_calculate_javascript_ast_similarity(self):
        """Test JavaScript AST similarity calculation"""
        predicted = "function test() { return 1; }"
        target = "function test() { return 42; }"
        
        similarity = self.evaluator._calculate_javascript_ast_similarity(predicted, target)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_calculate_complexity(self):
        """Test complexity calculation"""
        # Python code
        python_code = "def test(): if True: for i in range(10): pass"
        complexity = self.evaluator._calculate_complexity(python_code, "python")
        self.assertIsInstance(complexity, float)
        self.assertGreater(complexity, 1.0)
        
        # JavaScript code
        js_code = "function test() { if (true) { for (let i = 0; i < 10; i++) {} } }"
        complexity = self.evaluator._calculate_complexity(js_code, "javascript")
        self.assertIsInstance(complexity, float)
        self.assertGreater(complexity, 1.0)
    
    def test_calculate_pass_rate(self):
        """Test pass rate calculation"""
        # Valid Python code
        valid_python = "def test(): return 42"
        pass_rate = self.evaluator._calculate_pass_rate(valid_python, valid_python, "python")
        self.assertEqual(pass_rate, 1.0)
        
        # Invalid Python code
        invalid_python = "def test( return 42"
        pass_rate = self.evaluator._calculate_pass_rate(invalid_python, invalid_python, "python")
        self.assertEqual(pass_rate, 0.0)
        
        # Valid JavaScript code
        valid_js = "function test() { return 42; }"
        pass_rate = self.evaluator._calculate_pass_rate(valid_js, valid_js, "javascript")
        self.assertEqual(pass_rate, 1.0)


class TestContinualLearningEvaluator(unittest.TestCase):
    """Test ContinualLearningEvaluator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        mock_tokenizer = Mock()
        base_evaluator = ModelEvaluator(mock_tokenizer)
        self.cl_evaluator = ContinualLearningEvaluator(base_evaluator)
    
    def test_cl_evaluator_initialization(self):
        """Test continual learning evaluator initialization"""
        self.assertIsNotNone(self.cl_evaluator.evaluator)
    
    def test_calculate_continual_learning_metrics(self):
        """Test continual learning metrics calculation"""
        # Create mock evaluation results
        baseline_task1 = EvaluationResults(0.3, 0.2, 0.5, 0.4, 0.3, 2.0, 50, "python")
        baseline_task2 = EvaluationResults(0.2, 0.1, 0.3, 0.5, 0.2, 2.0, 50, "javascript")
        task1_after_task1 = EvaluationResults(0.7, 0.6, 0.8, 0.2, 0.7, 3.0, 50, "python")
        task2_after_task1 = EvaluationResults(0.3, 0.2, 0.4, 0.4, 0.3, 2.0, 50, "javascript")
        task1_after_task2 = EvaluationResults(0.6, 0.5, 0.7, 0.3, 0.6, 3.0, 50, "python")
        task2_after_task2 = EvaluationResults(0.6, 0.5, 0.7, 0.3, 0.6, 3.0, 50, "javascript")
        
        metrics = self.cl_evaluator.calculate_continual_learning_metrics(
            baseline_task1, baseline_task2,
            task1_after_task1, task2_after_task1,
            task1_after_task2, task2_after_task2
        )
        
        self.assertIn('forward_transfer', metrics)
        self.assertIn('backward_interference', metrics)
        self.assertIn('retention_score', metrics)
        self.assertIn('forgetting_rate', metrics)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(metrics['forward_transfer'], 0.0)
        self.assertGreaterEqual(metrics['backward_interference'], 0.0)
        self.assertGreaterEqual(metrics['retention_score'], 0.0)
        self.assertGreaterEqual(metrics['forgetting_rate'], 0.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_tokenizer = Mock()
        self.mock_model = Mock()
        self.mock_data = [{"func_code_string": "def test(): pass"}]
    
    @patch('utils.model_evaluator.ModelEvaluator')
    def test_evaluate_model_basic(self, mock_evaluator_class):
        """Test basic evaluation convenience function"""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_basic.return_value = (0.5, 0.8)
        mock_evaluator_class.return_value = mock_evaluator
        
        bleu, pass_rate = evaluate_model_basic(
            self.mock_model, self.mock_data, self.mock_tokenizer, "python", 10
        )
        
        self.assertEqual(bleu, 0.5)
        self.assertEqual(pass_rate, 0.8)
        mock_evaluator_class.assert_called_once_with(self.mock_tokenizer)
        mock_evaluator.evaluate_basic.assert_called_once_with(
            self.mock_model, self.mock_data, "python", 10
        )
    
    @patch('utils.model_evaluator.ModelEvaluator')
    def test_evaluate_model_comprehensive(self, mock_evaluator_class):
        """Test comprehensive evaluation convenience function"""
        mock_results = EvaluationResults(0.5, 0.4, 0.8, 0.3, 0.6, 5.0, 10, "python")
        mock_evaluator = Mock()
        mock_evaluator.evaluate_comprehensive.return_value = mock_results
        mock_evaluator_class.return_value = mock_evaluator
        
        results = evaluate_model_comprehensive(
            self.mock_model, self.mock_data, self.mock_tokenizer, "python", 10
        )
        
        self.assertEqual(results, mock_results)
        mock_evaluator_class.assert_called_once_with(self.mock_tokenizer)
        mock_evaluator.evaluate_comprehensive.assert_called_once_with(
            self.mock_model, self.mock_data, "python", 10
        )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_tokenizer = Mock()
        self.evaluator = ModelEvaluator(self.mock_tokenizer)
    
    def test_empty_data(self):
        """Test handling of empty data"""
        result = self.evaluator._extract_source_code("")
        self.assertEqual(result, "")
        
        result = self.evaluator._extract_source_code({})
        self.assertIsNone(result)
    
    def test_invalid_syntax(self):
        """Test handling of invalid syntax"""
        # Invalid Python
        similarity = self.evaluator._calculate_python_ast_similarity(
            "def test( invalid", "def test(): pass"
        )
        self.assertEqual(similarity, 0.0)
        
        # Invalid pass rate
        pass_rate = self.evaluator._calculate_pass_rate(
            "def test( invalid", "def test(): pass", "python"
        )
        self.assertEqual(pass_rate, 0.0)
    
    def test_empty_tokens(self):
        """Test handling of empty tokens"""
        self.mock_tokenizer.tokenize.return_value = []
        
        bleu = self.evaluator._calculate_bleu("", "def test(): pass")
        self.assertEqual(bleu, 0.0)
    
    def test_zero_division_protection(self):
        """Test protection against zero division"""
        # Test AST similarity with empty nodes
        similarity = self.evaluator._calculate_python_ast_similarity("", "")
        self.assertEqual(similarity, 0.0)
        
        # Test token similarity with empty tokens
        similarity = self.evaluator._calculate_token_similarity("", "")
        self.assertEqual(similarity, 0.0)


def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests() 