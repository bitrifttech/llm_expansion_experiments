"""
Model Evaluator Demo

This script demonstrates how to use the new ModelEvaluator utility class
to replace the evaluation logic in existing experiments.

It shows:
1. Basic usage of ModelEvaluator
2. Comprehensive evaluation with all metrics
3. Continual learning evaluation
4. How to migrate from existing evaluation code
5. Configuration options
"""

import sys
import os
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

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
from utils.data_loader import load_and_prepare_data


def demo_basic_evaluation():
    """Demonstrate basic model evaluation"""
    print("=== BASIC EVALUATION DEMO ===")
    
    # Load model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Load some sample data
    python_train, python_val, js_train, js_val = load_and_prepare_data(
        python_train_size=100,
        python_val_size=50,
        js_train_size=100,
        js_val_size=50,
        format_type="huggingface",
        seed=42
    )
    
    # Method 1: Using convenience function (backward compatible)
    print("\n1. Using convenience function:")
    bleu, pass_rate = evaluate_model_basic(model, python_val, tokenizer, "python", 10)
    print(f"   BLEU: {bleu:.4f}, Pass Rate: {pass_rate:.2%}")
    
    # Method 2: Using ModelEvaluator class
    print("\n2. Using ModelEvaluator class:")
    evaluator = ModelEvaluator(tokenizer)
    bleu, pass_rate = evaluator.evaluate_basic(model, python_val, "python", 10)
    print(f"   BLEU: {bleu:.4f}, Pass Rate: {pass_rate:.2%}")
    
    print("✓ Basic evaluation demo completed\n")


def demo_comprehensive_evaluation():
    """Demonstrate comprehensive evaluation with all metrics"""
    print("=== COMPREHENSIVE EVALUATION DEMO ===")
    
    # Load model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Load sample data
    python_train, python_val, js_train, js_val = load_and_prepare_data(
        python_train_size=100,
        python_val_size=50,
        js_train_size=100,
        js_val_size=50,
        format_type="huggingface",
        seed=42
    )
    
    # Create evaluator with custom configuration
    config = EvaluationConfig(
        max_length=256,  # Shorter for demo
        num_beams=2,     # Faster generation
        default_num_samples=10  # Small sample for demo
    )
    evaluator = ModelEvaluator(tokenizer, config)
    
    # Comprehensive evaluation
    print("\n1. Python evaluation:")
    python_results = evaluator.evaluate_comprehensive(model, python_val, "python", 10)
    print(f"   BLEU: {python_results.bleu:.4f}")
    print(f"   METEOR: {python_results.meteor:.4f}")
    print(f"   Pass Rate: {python_results.pass_rate:.2%}")
    print(f"   AST Similarity: {python_results.ast_similarity:.4f}")
    print(f"   Edit Distance: {python_results.edit_distance:.4f}")
    print(f"   Complexity: {python_results.complexity:.2f}")
    print(f"   Composite Score: {python_results.composite_score(config):.4f}")
    
    print("\n2. JavaScript evaluation:")
    js_results = evaluator.evaluate_comprehensive(model, js_val, "javascript", 10)
    print(f"   BLEU: {js_results.bleu:.4f}")
    print(f"   METEOR: {js_results.meteor:.4f}")
    print(f"   Pass Rate: {js_results.pass_rate:.2%}")
    print(f"   AST Similarity: {js_results.ast_similarity:.4f}")
    print(f"   Edit Distance: {js_results.edit_distance:.4f}")
    print(f"   Complexity: {js_results.complexity:.2f}")
    print(f"   Composite Score: {js_results.composite_score(config):.4f}")
    
    # Export results
    print("\n3. Exporting results:")
    python_dict = python_results.to_dict()
    print(f"   Python results exported: {len(python_dict)} metrics")
    
    print("✓ Comprehensive evaluation demo completed\n")


def demo_continual_learning_evaluation():
    """Demonstrate continual learning evaluation"""
    print("=== CONTINUAL LEARNING EVALUATION DEMO ===")
    
    # Load model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # For demo, we'll use the same model for all stages
    # In real experiments, these would be different model states
    base_model = T5ForConditionalGeneration.from_pretrained(model_name)
    model_after_task1 = T5ForConditionalGeneration.from_pretrained(model_name)
    model_after_task2 = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Load sample data
    python_train, python_val, js_train, js_val = load_and_prepare_data(
        python_train_size=100,
        python_val_size=50,
        js_train_size=100,
        js_val_size=50,
        format_type="huggingface",
        seed=42
    )
    
    # Create continual learning evaluator
    base_evaluator = ModelEvaluator(tokenizer, EvaluationConfig(default_num_samples=10))
    cl_evaluator = ContinualLearningEvaluator(base_evaluator)
    
    # Run complete continual learning evaluation
    print("\n1. Running complete continual learning evaluation...")
    results = cl_evaluator.evaluate_continual_learning_experiment(
        model_before_training=base_model,
        model_after_task1=model_after_task1,
        model_after_task2=model_after_task2,
        task1_data=python_val,
        task2_data=js_val,
        task1_language="python",
        task2_language="javascript",
        num_samples=10
    )
    
    # Display continual learning metrics
    cl_metrics = results['continual_learning_metrics']
    print(f"\n2. Continual Learning Metrics:")
    print(f"   Forward Transfer: {cl_metrics['forward_transfer']:+.4f}")
    print(f"   Backward Interference: {cl_metrics['backward_interference']:+.4f}")
    print(f"   Retention Score: {cl_metrics['retention_score']:.4f}")
    print(f"   Forgetting Rate: {cl_metrics['forgetting_rate']:.2%}")
    print(f"   Task 1 Retention: {cl_metrics['task1_retention']:.4f}")
    print(f"   Task 2 Improvement: {cl_metrics['task2_improvement']:.4f}")
    
    print("✓ Continual learning evaluation demo completed\n")


def demo_migration_example():
    """Show how to migrate from existing evaluation code"""
    print("=== MIGRATION EXAMPLE ===")
    
    print("OLD CODE (from experiments):")
    print("""
    def evaluate_model(self, model, data, num_samples: int = 100, language: str = None):
        model.eval()
        bleu_scores = []
        pass_scores = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(data))):
                # ... complex evaluation logic ...
                bleu_scores.append(bleu_score)
                pass_scores.append(pass_score)
        
        return {
            'bleu': np.mean(bleu_scores),
            'pass_rate': np.mean(pass_scores)
        }
    """)
    
    print("\nNEW CODE (using ModelEvaluator):")
    print("""
    from utils.model_evaluator import ModelEvaluator
    
    def evaluate_model(self, model, data, num_samples: int = 100, language: str = None):
        evaluator = ModelEvaluator(self.tokenizer)
        results = evaluator.evaluate_comprehensive(model, data, language, num_samples)
        return results.to_dict()  # Returns all metrics, not just BLEU and pass rate
    """)
    
    print("\nBENEFITS:")
    print("✓ Consistent evaluation across all experiments")
    print("✓ More metrics (METEOR, AST similarity, complexity, etc.)")
    print("✓ Better error handling and edge case management")
    print("✓ Configurable generation parameters")
    print("✓ Language auto-detection")
    print("✓ Continual learning metrics built-in")
    print("✓ Backward compatibility with existing code")
    
    print("✓ Migration example completed\n")


def demo_configuration_options():
    """Demonstrate configuration options"""
    print("=== CONFIGURATION OPTIONS DEMO ===")
    
    # Default configuration
    default_config = EvaluationConfig()
    print("1. Default Configuration:")
    print(f"   Max Length: {default_config.max_length}")
    print(f"   Num Beams: {default_config.num_beams}")
    print(f"   Temperature: {default_config.temperature}")
    print(f"   Default Samples: {default_config.default_num_samples}")
    print(f"   Input Fraction: {default_config.input_fraction}")
    
    # Custom configuration for fast evaluation
    fast_config = EvaluationConfig(
        max_length=128,
        num_beams=1,
        do_sample=False,
        default_num_samples=50,
        input_fraction=0.3
    )
    print("\n2. Fast Evaluation Configuration:")
    print(f"   Max Length: {fast_config.max_length}")
    print(f"   Num Beams: {fast_config.num_beams}")
    print(f"   Do Sample: {fast_config.do_sample}")
    print(f"   Default Samples: {fast_config.default_num_samples}")
    print(f"   Input Fraction: {fast_config.input_fraction}")
    
    # Custom configuration for high-quality evaluation
    quality_config = EvaluationConfig(
        max_length=1024,
        num_beams=5,
        temperature=0.5,
        default_num_samples=200,
        bleu_weight=0.4,
        ast_similarity_weight=0.3
    )
    print("\n3. High-Quality Evaluation Configuration:")
    print(f"   Max Length: {quality_config.max_length}")
    print(f"   Num Beams: {quality_config.num_beams}")
    print(f"   Temperature: {quality_config.temperature}")
    print(f"   Default Samples: {quality_config.default_num_samples}")
    print(f"   BLEU Weight: {quality_config.bleu_weight}")
    print(f"   AST Similarity Weight: {quality_config.ast_similarity_weight}")
    
    print("✓ Configuration options demo completed\n")


def main():
    """Run all demos"""
    print("MODEL EVALUATOR DEMONSTRATION")
    print("=" * 50)
    
    try:
        demo_basic_evaluation()
        demo_comprehensive_evaluation()
        demo_continual_learning_evaluation()
        demo_migration_example()
        demo_configuration_options()
        
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\nNext Steps:")
        print("1. Replace evaluation code in experiments with ModelEvaluator")
        print("2. Use EvaluationConfig to customize evaluation parameters")
        print("3. Use ContinualLearningEvaluator for continual learning experiments")
        print("4. Export results using .to_dict() for consistent data format")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 