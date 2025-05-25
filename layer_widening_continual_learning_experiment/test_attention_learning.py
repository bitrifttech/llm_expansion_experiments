import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
import os
from copy import deepcopy

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main file
from attention_head_expansion_continual_learning import AttentionHeadExpansionContinualLearner, ExpandedMultiHeadAttention, log_message

def test_attention_head_learning():
    """Test that new attention heads are actually learning and contributing"""
    log_message("=== TESTING ATTENTION HEAD LEARNING AND CONTRIBUTION ===")
    
    # Initialize model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cpu"  # Use CPU for testing
    
    # Initialize learner
    learner = AttentionHeadExpansionContinualLearner(model_name, tokenizer, device, num_new_heads=1)
    learner.prepare_model()
    
    log_message("Learner initialized successfully")
    
    # Create training data with clear patterns
    train_data = [
        {'input': 'Generate Python code: add two numbers', 'target': 'def add(a, b): return a + b'},
        {'input': 'Generate Python code: multiply two numbers', 'target': 'def multiply(a, b): return a * b'},
        {'input': 'Generate Python code: subtract two numbers', 'target': 'def subtract(a, b): return a - b'},
        {'input': 'Generate Python code: divide two numbers', 'target': 'def divide(a, b): return a / b'},
        {'input': 'Generate Python code: find maximum', 'target': 'def max_val(a, b): return max(a, b)'},
        {'input': 'Generate Python code: find minimum', 'target': 'def min_val(a, b): return min(a, b)'},
    ]
    
    # Test data for evaluation
    test_data = [
        {'input': 'Generate Python code: calculate power', 'target': 'def power(a, b): return a ** b'},
        {'input': 'Generate Python code: modulo operation', 'target': 'def mod(a, b): return a % b'},
    ]
    
    log_message("Created training and test datasets")
    
    # Step 1: Capture initial state of attention heads
    log_message("\n=== STEP 1: CAPTURING INITIAL STATE ===")
    # We need to create the expanded model first to capture initial stats
    from attention_head_expansion_continual_learning import expand_model_attention_heads
    temp_expanded_model = expand_model_attention_heads(learner.base_model, learner.num_new_heads)
    initial_stats = capture_attention_stats(temp_expanded_model)
    log_message(f"Initial attention head stats captured for {len(initial_stats)} layers")
    
    # Step 2: Test initial inference capability
    log_message("\n=== STEP 2: TESTING INITIAL INFERENCE ===")
    initial_outputs = test_inference_capability(learner.base_model, tokenizer, test_data, "BEFORE training")
    
    # Step 3: Train the model and track learning
    log_message("\n=== STEP 3: TRAINING WITH LEARNING TRACKING ===")
    expanded_model = train_with_learning_tracking(learner, train_data, temp_expanded_model)
    
    # Step 4: Capture final state of attention heads
    log_message("\n=== STEP 4: CAPTURING FINAL STATE ===")
    final_stats = capture_attention_stats(expanded_model)
    log_message(f"Final attention head stats captured for {len(final_stats)} layers")
    
    # Step 5: Test final inference capability
    log_message("\n=== STEP 5: TESTING FINAL INFERENCE ===")
    final_outputs = test_inference_capability(expanded_model, tokenizer, test_data, "AFTER training")
    
    # Step 6: Analyze learning and contribution
    log_message("\n=== STEP 6: ANALYZING LEARNING AND CONTRIBUTION ===")
    learning_analysis = analyze_learning_progress(initial_stats, final_stats)
    contribution_analysis = analyze_inference_contribution(expanded_model, tokenizer, test_data)
    output_comparison = compare_outputs(initial_outputs, final_outputs)
    
    # Step 7: Generate report
    log_message("\n=== STEP 7: GENERATING LEARNING REPORT ===")
    generate_learning_report(learning_analysis, contribution_analysis, output_comparison)
    
    return True

def capture_attention_stats(model):
    """Capture statistics of all attention heads"""
    stats = {}
    
    for name, module in model.named_modules():
        if isinstance(module, ExpandedMultiHeadAttention):
            module_stats = module.get_training_stats()
            stats[name] = {
                'gate_value': module_stats['gate_value'],
                'weight_norms': {
                    'q': module_stats['new_q_weight_norm'],
                    'k': module_stats['new_k_weight_norm'],
                    'v': module_stats['new_v_weight_norm'],
                    'o': module_stats['new_o_weight_norm']
                },
                'weights_snapshot': {
                    'q': module.new_q_proj.weight.data.clone(),
                    'k': module.new_k_proj.weight.data.clone(),
                    'v': module.new_v_proj.weight.data.clone(),
                    'o': module.new_o_proj.weight.data.clone()
                }
            }
    
    return stats

def test_inference_capability(model, tokenizer, test_data, phase_name):
    """Test inference capability and capture outputs"""
    log_message(f"Testing inference capability {phase_name}...")
    
    outputs = []
    model.eval()
    
    with torch.no_grad():
        for i, item in enumerate(test_data):
            input_text = item['input']
            target_text = item['target']
            
            # Tokenize input
            input_encoding = tokenizer(
                input_text, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            )
            
            try:
                # Generate prediction
                generated_ids = model.generate(
                    input_ids=input_encoding.input_ids,
                    attention_mask=input_encoding.attention_mask,
                    max_length=100,
                    num_beams=2,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode prediction
                predicted_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                outputs.append({
                    'input': input_text,
                    'target': target_text,
                    'predicted': predicted_text,
                    'input_length': len(input_encoding.input_ids[0]),
                    'output_length': len(generated_ids[0])
                })
                
                log_message(f"  Sample {i+1}: '{predicted_text[:50]}...'")
                
            except Exception as e:
                log_message(f"  Sample {i+1}: FAILED - {e}")
                outputs.append({
                    'input': input_text,
                    'target': target_text,
                    'predicted': f"ERROR: {e}",
                    'input_length': 0,
                    'output_length': 0
                })
    
    return outputs

def train_with_learning_tracking(learner, train_data, temp_expanded_model):
    """Train model while tracking learning progress"""
    log_message("Training with detailed learning tracking...")
    
    # Get trainable parameters
    trainable_params = [p for p in temp_expanded_model.parameters() if p.requires_grad]
    log_message(f"Training {len(trainable_params)} parameter groups")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.01)
    
    # Collect expanded attention modules
    expanded_attentions = []
    for name, module in temp_expanded_model.named_modules():
        if isinstance(module, ExpandedMultiHeadAttention):
            expanded_attentions.append((name, module))
    
    log_message(f"Found {len(expanded_attentions)} ExpandedMultiHeadAttention modules")
    
    # Training loop with detailed tracking
    temp_expanded_model.train()
    batch_size = 2
    epochs = 3  # More epochs for better learning
    
    learning_progress = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        valid_batches = 0
        
        log_message(f"\n--- EPOCH {epoch+1}/{epochs} ---")
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Prepare batch
            input_texts = [item['input'] for item in batch]
            target_texts = [item['target'] for item in batch]
            
            # Tokenize
            input_encodings = tokenizer(
                input_texts, 
                truncation=True, 
                padding=True, 
                max_length=512, 
                return_tensors="pt"
            )
            
            target_encodings = tokenizer(
                target_texts, 
                truncation=True, 
                padding=True, 
                max_length=512, 
                return_tensors="pt"
            )
            
            try:
                # Forward pass
                outputs = temp_expanded_model(
                    input_ids=input_encodings.input_ids,
                    attention_mask=input_encodings.attention_mask,
                    labels=target_encodings.input_ids
                )
                
                loss = outputs.loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    log_message(f"  Batch {i//batch_size + 1}: Invalid loss, skipping")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
                
                # Track learning progress every batch
                if valid_batches % 1 == 0:  # Every batch for detailed tracking
                    batch_stats = {}
                    for name, attention in expanded_attentions[:3]:  # Track first 3 layers
                        stats = attention.get_training_stats()
                        batch_stats[name] = {
                            'gate': stats['gate_value'],
                            'q_grad': stats['new_q_grad_norm'],
                            'k_grad': stats['new_k_grad_norm'],
                            'v_grad': stats['new_v_grad_norm'],
                            'weight_norms': [
                                stats['new_q_weight_norm'],
                                stats['new_k_weight_norm'],
                                stats['new_v_weight_norm']
                            ]
                        }
                    
                    learning_progress.append({
                        'epoch': epoch + 1,
                        'batch': valid_batches,
                        'loss': loss.item(),
                        'attention_stats': batch_stats
                    })
                    
                    # Log progress
                    sample_stats = list(batch_stats.values())[0] if batch_stats else {}
                    log_message(f"  Batch {valid_batches}: Loss={loss.item():.4f}, "
                              f"Gate={sample_stats.get('gate', 0):.6f}, "
                              f"Grads=[Q:{sample_stats.get('q_grad', 0):.6f}, "
                              f"K:{sample_stats.get('k_grad', 0):.6f}, "
                              f"V:{sample_stats.get('v_grad', 0):.6f}]")
                
            except Exception as e:
                log_message(f"  Batch {i//batch_size + 1}: Training error - {e}")
                continue
        
        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
        log_message(f"Epoch {epoch+1} completed: Avg Loss={avg_loss:.4f}, Valid Batches={valid_batches}")
    
    # Save learning progress for analysis
    learner.learning_progress = learning_progress
    
    return temp_expanded_model

def analyze_learning_progress(initial_stats, final_stats):
    """Analyze how much the attention heads learned"""
    log_message("Analyzing learning progress...")
    
    analysis = {
        'layers_analyzed': 0,
        'gate_changes': [],
        'weight_changes': [],
        'total_parameter_change': 0.0
    }
    
    for layer_name in initial_stats:
        if layer_name in final_stats:
            initial = initial_stats[layer_name]
            final = final_stats[layer_name]
            
            # Gate value change
            gate_change = final['gate_value'] - initial['gate_value']
            analysis['gate_changes'].append(gate_change)
            
            # Weight norm changes
            weight_change = 0.0
            for weight_type in ['q', 'k', 'v', 'o']:
                initial_norm = initial['weight_norms'][weight_type]
                final_norm = final['weight_norms'][weight_type]
                weight_change += abs(final_norm - initial_norm)
            
            analysis['weight_changes'].append(weight_change)
            
            # Calculate actual parameter changes
            param_change = 0.0
            for weight_type in ['q', 'k', 'v', 'o']:
                initial_weights = initial['weights_snapshot'][weight_type]
                final_weights = final['weights_snapshot'][weight_type]
                param_change += torch.norm(final_weights - initial_weights).item()
            
            analysis['total_parameter_change'] += param_change
            analysis['layers_analyzed'] += 1
            
            log_message(f"  {layer_name}: Gate Δ={gate_change:.6f}, Weight Δ={weight_change:.4f}, Param Δ={param_change:.4f}")
    
    # Summary statistics
    if analysis['gate_changes']:
        analysis['avg_gate_change'] = np.mean(analysis['gate_changes'])
        analysis['max_gate_change'] = np.max(analysis['gate_changes'])
        analysis['avg_weight_change'] = np.mean(analysis['weight_changes'])
        analysis['max_weight_change'] = np.max(analysis['weight_changes'])
    
    return analysis

def analyze_inference_contribution(model, tokenizer, test_data):
    """Analyze how much new attention heads contribute to inference"""
    log_message("Analyzing inference contribution...")
    
    contribution_analysis = {
        'with_new_heads': [],
        'without_new_heads': [],
        'contribution_scores': []
    }
    
    model.eval()
    
    for item in test_data:
        input_text = item['input']
        
        # Tokenize input
        input_encoding = tokenizer(
            input_text, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            try:
                # Test with new heads enabled (normal mode)
                outputs_with = model.generate(
                    input_ids=input_encoding.input_ids,
                    attention_mask=input_encoding.attention_mask,
                    max_length=50,
                    num_beams=1,  # Greedy for consistency
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                text_with = tokenizer.decode(outputs_with[0], skip_special_tokens=True)
                
                # Test with new heads disabled (set gate to 0)
                # Temporarily disable new heads
                original_gates = []
                for name, module in model.named_modules():
                    if isinstance(module, ExpandedMultiHeadAttention):
                        original_gates.append(module.gate.data.clone())
                        module.gate.data.fill_(-20.0)  # Very negative = ~0 contribution
                
                outputs_without = model.generate(
                    input_ids=input_encoding.input_ids,
                    attention_mask=input_encoding.attention_mask,
                    max_length=50,
                    num_beams=1,  # Greedy for consistency
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                text_without = tokenizer.decode(outputs_without[0], skip_special_tokens=True)
                
                # Restore original gates
                gate_idx = 0
                for name, module in model.named_modules():
                    if isinstance(module, ExpandedMultiHeadAttention):
                        module.gate.data = original_gates[gate_idx]
                        gate_idx += 1
                
                # Calculate contribution score (simple text difference)
                contribution_score = 1.0 if text_with != text_without else 0.0
                
                contribution_analysis['with_new_heads'].append(text_with)
                contribution_analysis['without_new_heads'].append(text_without)
                contribution_analysis['contribution_scores'].append(contribution_score)
                
                log_message(f"  Input: {input_text[:50]}...")
                log_message(f"    With new heads: {text_with[:50]}...")
                log_message(f"    Without new heads: {text_without[:50]}...")
                log_message(f"    Contribution score: {contribution_score}")
                
            except Exception as e:
                log_message(f"  Error analyzing contribution: {e}")
                contribution_analysis['contribution_scores'].append(0.0)
    
    return contribution_analysis

def compare_outputs(initial_outputs, final_outputs):
    """Compare outputs before and after training"""
    log_message("Comparing outputs before and after training...")
    
    comparison = {
        'improved_samples': 0,
        'degraded_samples': 0,
        'unchanged_samples': 0,
        'details': []
    }
    
    for i, (initial, final) in enumerate(zip(initial_outputs, final_outputs)):
        initial_pred = initial['predicted']
        final_pred = final['predicted']
        target = initial['target']
        
        # Simple improvement metric: closer to target
        initial_similarity = calculate_similarity(initial_pred, target)
        final_similarity = calculate_similarity(final_pred, target)
        
        if final_similarity > initial_similarity:
            comparison['improved_samples'] += 1
            status = "IMPROVED"
        elif final_similarity < initial_similarity:
            comparison['degraded_samples'] += 1
            status = "DEGRADED"
        else:
            comparison['unchanged_samples'] += 1
            status = "UNCHANGED"
        
        comparison['details'].append({
            'sample': i + 1,
            'status': status,
            'initial_similarity': initial_similarity,
            'final_similarity': final_similarity,
            'improvement': final_similarity - initial_similarity
        })
        
        log_message(f"  Sample {i+1}: {status} (Δ={final_similarity - initial_similarity:.3f})")
    
    return comparison

def calculate_similarity(text1, text2):
    """Calculate simple similarity between two texts"""
    # Simple word overlap similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def generate_learning_report(learning_analysis, contribution_analysis, output_comparison):
    """Generate comprehensive learning report"""
    log_message("\n" + "="*60)
    log_message("ATTENTION HEAD LEARNING AND CONTRIBUTION REPORT")
    log_message("="*60)
    
    # Learning Progress Report
    log_message("\n📈 LEARNING PROGRESS:")
    log_message(f"  Layers Analyzed: {learning_analysis['layers_analyzed']}")
    if 'avg_gate_change' in learning_analysis:
        log_message(f"  Average Gate Change: {learning_analysis['avg_gate_change']:.6f}")
        log_message(f"  Maximum Gate Change: {learning_analysis['max_gate_change']:.6f}")
        log_message(f"  Average Weight Change: {learning_analysis['avg_weight_change']:.4f}")
        log_message(f"  Total Parameter Change: {learning_analysis['total_parameter_change']:.4f}")
    
    # Contribution Analysis Report
    log_message("\n🎯 INFERENCE CONTRIBUTION:")
    if contribution_analysis['contribution_scores']:
        avg_contribution = np.mean(contribution_analysis['contribution_scores'])
        total_samples = len(contribution_analysis['contribution_scores'])
        contributing_samples = sum(contribution_analysis['contribution_scores'])
        log_message(f"  Samples with measurable contribution: {contributing_samples}/{total_samples}")
        log_message(f"  Average contribution score: {avg_contribution:.3f}")
    
    # Output Comparison Report
    log_message("\n📊 OUTPUT QUALITY CHANGES:")
    total_samples = (output_comparison['improved_samples'] + 
                    output_comparison['degraded_samples'] + 
                    output_comparison['unchanged_samples'])
    if total_samples > 0:
        log_message(f"  Improved samples: {output_comparison['improved_samples']}/{total_samples} "
                   f"({100*output_comparison['improved_samples']/total_samples:.1f}%)")
        log_message(f"  Degraded samples: {output_comparison['degraded_samples']}/{total_samples} "
                   f"({100*output_comparison['degraded_samples']/total_samples:.1f}%)")
        log_message(f"  Unchanged samples: {output_comparison['unchanged_samples']}/{total_samples} "
                   f"({100*output_comparison['unchanged_samples']/total_samples:.1f}%)")
    
    # Overall Assessment
    log_message("\n🔍 OVERALL ASSESSMENT:")
    
    # Check if learning occurred
    learning_occurred = False
    if 'avg_gate_change' in learning_analysis:
        if abs(learning_analysis['avg_gate_change']) > 1e-6 or learning_analysis['total_parameter_change'] > 1e-3:
            learning_occurred = True
    
    # Check if contribution is measurable
    contribution_measurable = False
    if contribution_analysis['contribution_scores']:
        if np.mean(contribution_analysis['contribution_scores']) > 0:
            contribution_measurable = True
    
    # Check if outputs improved
    outputs_improved = False
    if total_samples > 0:
        if output_comparison['improved_samples'] > output_comparison['degraded_samples']:
            outputs_improved = True
    
    log_message(f"  ✅ Learning Detected: {'YES' if learning_occurred else 'NO'}")
    log_message(f"  ✅ Contribution Measurable: {'YES' if contribution_measurable else 'NO'}")
    log_message(f"  ✅ Output Quality Improved: {'YES' if outputs_improved else 'NO'}")
    
    # Final verdict
    if learning_occurred and contribution_measurable:
        log_message("\n🎉 VERDICT: New attention heads are LEARNING and CONTRIBUTING!")
    elif learning_occurred:
        log_message("\n⚠️  VERDICT: New attention heads are learning but contribution is minimal.")
    else:
        log_message("\n❌ VERDICT: New attention heads are NOT learning effectively.")
    
    log_message("="*60)

if __name__ == "__main__":
    log_message("Starting comprehensive attention head learning test...")
    
    # Import tokenizer here to avoid issues
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    
    if test_attention_head_learning():
        log_message("Attention head learning test completed successfully!")
    else:
        log_message("Attention head learning test failed!")
        exit(1) 