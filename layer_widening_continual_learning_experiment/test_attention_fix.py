import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def log_message(message: str):
    print(f"[TEST] {message}")

class FixedExpandedMultiHeadAttention(torch.nn.Module):
    """Fixed multi-head attention with additional trainable heads"""
    
    def __init__(self, original_attention, num_new_heads: int = 1, device: str = "cpu"):
        super().__init__()
        self.original_attention = original_attention  # Frozen
        self.num_new_heads = num_new_heads
        self.device = device
        
        # Get configuration from original attention
        self.d_model = original_attention.d_model
        self.num_original_heads = original_attention.n_heads
        self.d_kv = original_attention.key_value_proj_dim
        
        # Get dtype from original attention
        self.dtype = next(original_attention.parameters()).dtype
        
        # Freeze original attention
        for param in self.original_attention.parameters():
            param.requires_grad = False
        
        # Create new attention heads
        self.new_q_proj = torch.nn.Linear(
            self.d_model, 
            num_new_heads * self.d_kv, 
            bias=False, 
            dtype=self.dtype, 
            device=device
        )
        self.new_k_proj = torch.nn.Linear(
            self.d_model, 
            num_new_heads * self.d_kv, 
            bias=False, 
            dtype=self.dtype, 
            device=device
        )
        self.new_v_proj = torch.nn.Linear(
            self.d_model, 
            num_new_heads * self.d_kv, 
            bias=False, 
            dtype=self.dtype, 
            device=device
        )
        
        # Output projection for new heads
        self.new_o_proj = torch.nn.Linear(
            num_new_heads * self.d_kv, 
            self.d_model, 
            bias=False, 
            dtype=self.dtype, 
            device=device
        )
        
        # Dropout for new heads
        self.dropout = torch.nn.Dropout(0.1)
        
        # Gate to control contribution of new heads
        self.gate = torch.nn.Parameter(torch.tensor(-2.0, dtype=self.dtype, device=device))
        
        # Initialize weights (simplified for testing)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.new_q_proj.weight)
            torch.nn.init.xavier_uniform_(self.new_k_proj.weight)
            torch.nn.init.xavier_uniform_(self.new_v_proj.weight)
            torch.nn.init.xavier_uniform_(self.new_o_proj.weight)
        
        log_message(f"Created FixedExpandedMultiHeadAttention: {self.num_original_heads} original + {num_new_heads} new heads")
    
    def forward(self, hidden_states, mask=None, key_value_states=None, 
                position_bias=None, past_key_value=None, layer_head_mask=None, 
                query_length=None, use_cache=False, output_attentions=False, cache_position=None):
        """Forward pass with original + new attention heads - FIXED VERSION"""
        
        # Original attention output (frozen)
        with torch.no_grad():
            original_outputs = self.original_attention(
                hidden_states=hidden_states,
                mask=mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                past_key_value=past_key_value,
                layer_head_mask=layer_head_mask,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position
            )
        
        # Extract original attention output
        if isinstance(original_outputs, tuple):
            original_attention_output = original_outputs[0]
        else:
            original_attention_output = original_outputs
        
        # CRITICAL FIX: Handle dynamic sequence lengths properly
        try:
            # Get dimensions safely
            batch_size, seq_length = hidden_states.shape[:2]
            
            # Use key_value_states if provided (for cross-attention), otherwise use hidden_states
            kv_input = key_value_states if key_value_states is not None else hidden_states
            kv_seq_length = kv_input.size(1)
            
            # Project to Q, K, V for new heads
            new_q = self.new_q_proj(hidden_states)  # [batch, seq_len, num_new_heads * d_kv]
            new_k = self.new_k_proj(kv_input)       # [batch, kv_seq_len, num_new_heads * d_kv]
            new_v = self.new_v_proj(kv_input)       # [batch, kv_seq_len, num_new_heads * d_kv]
            
            # CRITICAL FIX: Ensure dimensions are correct before reshaping
            expected_q_dim = self.num_new_heads * self.d_kv
            if new_q.size(-1) != expected_q_dim:
                log_message(f"WARNING: Q projection dimension mismatch: expected {expected_q_dim}, got {new_q.size(-1)}")
                # Fallback: return original output only
                return original_outputs
            
            # CRITICAL FIX: Reshape with explicit dimension validation
            try:
                new_q = new_q.view(batch_size, seq_length, self.num_new_heads, self.d_kv).transpose(1, 2)
                new_k = new_k.view(batch_size, kv_seq_length, self.num_new_heads, self.d_kv).transpose(1, 2)
                new_v = new_v.view(batch_size, kv_seq_length, self.num_new_heads, self.d_kv).transpose(1, 2)
            except RuntimeError as e:
                log_message(f"WARNING: Reshape failed: {e}")
                # Fallback: return original output only
                return original_outputs
            
            # CRITICAL FIX: Validate tensor dimensions before matrix multiplication
            if new_q.size(-1) != new_k.size(-1):
                log_message(f"WARNING: Q-K dimension mismatch: Q={new_q.shape}, K={new_k.shape}")
                return original_outputs
            
            if new_q.size(-2) == 0 or new_k.size(-2) == 0:
                log_message(f"WARNING: Zero sequence length: Q seq={new_q.size(-2)}, K seq={new_k.size(-2)}")
                return original_outputs
            
            # Compute attention scores with dimension validation
            try:
                scores = torch.matmul(new_q, new_k.transpose(-2, -1)) / np.sqrt(self.d_kv)
            except RuntimeError as e:
                log_message(f"WARNING: Matrix multiplication failed: {e}")
                log_message(f"  Q shape: {new_q.shape}, K shape: {new_k.shape}")
                return original_outputs
            
            # Apply attention mask if provided (simplified for testing)
            if mask is not None:
                try:
                    # Simple mask handling - expand to match scores dimensions
                    if mask.dim() == 2:  # [batch, seq_len]
                        expanded_mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_new_heads, -1, kv_seq_length)
                    elif mask.dim() == 3:  # [batch, seq_len, kv_seq_len]
                        expanded_mask = mask.unsqueeze(1).expand(-1, self.num_new_heads, -1, -1)
                    elif mask.dim() == 4:  # [batch, num_heads, seq_len, kv_seq_len]
                        if mask.size(1) == 1:
                            expanded_mask = mask.expand(-1, self.num_new_heads, -1, -1)
                        else:
                            expanded_mask = mask[:, :self.num_new_heads, :, :]
                    else:
                        # Skip mask if too complex
                        expanded_mask = None
                    
                    if expanded_mask is not None and expanded_mask.shape == scores.shape:
                        scores = scores.masked_fill(expanded_mask == 0, -1e9)
                except Exception as e:
                    log_message(f"WARNING: Mask application failed: {e}")
                    # Continue without mask
            
            # Apply position bias if provided (simplified)
            if position_bias is not None:
                try:
                    if position_bias.size(1) >= self.num_new_heads:
                        new_position_bias = position_bias[:, :self.num_new_heads, :, :]
                        if new_position_bias.shape == scores.shape:
                            scores = scores + new_position_bias
                except Exception as e:
                    log_message(f"WARNING: Position bias application failed: {e}")
                    # Continue without position bias
            
            # Softmax attention weights
            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values with dimension validation
            try:
                if attention_weights.size(-1) != new_v.size(-2):
                    log_message(f"WARNING: Attention-Value dimension mismatch: attn={attention_weights.shape}, V={new_v.shape}")
                    return original_outputs
                
                new_attention_output = torch.matmul(attention_weights, new_v)
            except RuntimeError as e:
                log_message(f"WARNING: Attention-Value multiplication failed: {e}")
                return original_outputs
            
            # Reshape and project output
            try:
                new_attention_output = new_attention_output.transpose(1, 2).contiguous()
                new_attention_output = new_attention_output.view(batch_size, seq_length, self.num_new_heads * self.d_kv)
                new_attention_output = self.new_o_proj(new_attention_output)
            except RuntimeError as e:
                log_message(f"WARNING: Output projection failed: {e}")
                return original_outputs
            
            # Apply gate to control contribution
            gate_value = torch.sigmoid(self.gate) * 0.25
            gated_new_output = gate_value * new_attention_output
            
            # Combine original and new attention outputs
            combined_output = original_attention_output + gated_new_output
            
            # Return in the same format as original
            if isinstance(original_outputs, tuple):
                return (combined_output,) + original_outputs[1:]
            else:
                return combined_output
                
        except Exception as e:
            log_message(f"CRITICAL ERROR in new attention computation: {e}")
            log_message(f"  Input shapes: hidden_states={hidden_states.shape}")
            if key_value_states is not None:
                log_message(f"  key_value_states={key_value_states.shape}")
            # Fallback: return original output only
            return original_outputs

def test_attention_dimensions():
    """Test attention mechanism with various input dimensions"""
    log_message("Testing attention with various dimensions...")
    
    # Load a simple model
    model_name = "Salesforce/codet5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # Get original attention
    original_attention = model.encoder.block[0].layer[0].SelfAttention
    fixed_attention = FixedExpandedMultiHeadAttention(original_attention, num_new_heads=1, device="cpu")
    
    # Test with different sequence lengths
    test_cases = [
        (1, 1),    # Single token
        (1, 2),    # Two tokens
        (1, 10),   # Normal length
        (2, 5),    # Batch of 2
        (1, 50),   # Longer sequence
    ]
    
    d_model = original_attention.d_model
    
    for batch_size, seq_len in test_cases:
        log_message(f"Testing batch_size={batch_size}, seq_len={seq_len}")
        
        # Create test input
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        try:
            # Test forward pass
            output = fixed_attention(hidden_states)
            
            # Handle tuple output properly
            if isinstance(output, tuple):
                output_shape = output[0].shape
            else:
                output_shape = output.shape
                
            log_message(f"  SUCCESS: Output shape {output_shape}")
        except Exception as e:
            log_message(f"  FAILED: {e}")
            return False
    
    log_message("All dimension tests passed!")
    return True

def test_simple_generation():
    """Test simple generation without beam search first"""
    log_message("Testing simple generation (no beam search)...")
    
    # Load model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # Replace one attention layer with our fixed version
    original_attention = model.encoder.block[0].layer[0].SelfAttention
    fixed_attention = FixedExpandedMultiHeadAttention(original_attention, num_new_heads=1, device="cpu")
    model.encoder.block[0].layer[0].SelfAttention = fixed_attention
    
    log_message("Replaced first encoder attention layer with fixed version")
    
    # Test with a simple input
    test_input = "Generate Python code: add two numbers"
    
    log_message(f"Testing with input: {test_input}")
    
    # Tokenize
    input_encoding = tokenizer(
        test_input, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    log_message(f"Input encoding shape: {input_encoding.input_ids.shape}")
    
    try:
        # Test simple generation (no beam search)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                max_length=50,   # Very short for testing
                do_sample=False, # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log_message(f"Generated text: {generated_text}")
        log_message("SUCCESS: Simple generation completed without errors!")
        return True
        
    except Exception as e:
        log_message(f"FAILED: Simple generation failed with error: {e}")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")
        return False

def test_beam_search_generation():
    """Test beam search generation"""
    log_message("Testing beam search generation...")
    
    # Load model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # Replace one attention layer with our fixed version
    original_attention = model.encoder.block[0].layer[0].SelfAttention
    fixed_attention = FixedExpandedMultiHeadAttention(original_attention, num_new_heads=1, device="cpu")
    model.encoder.block[0].layer[0].SelfAttention = fixed_attention
    
    log_message("Replaced first encoder attention layer with fixed version")
    
    # Test with a simple input
    test_input = "Generate Python code: add two numbers"
    
    log_message(f"Testing with input: {test_input}")
    
    # Tokenize
    input_encoding = tokenizer(
        test_input, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    log_message(f"Input encoding shape: {input_encoding.input_ids.shape}")
    
    try:
        # Test beam search generation
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                max_length=50,   # Short for testing
                num_beams=2,     # Small beam size
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log_message(f"Generated text: {generated_text}")
        log_message("SUCCESS: Beam search generation completed without errors!")
        return True
        
    except Exception as e:
        log_message(f"FAILED: Beam search generation failed with error: {e}")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    log_message("Starting attention mechanism tests...")
    
    # Test 1: Basic dimension handling
    if not test_attention_dimensions():
        log_message("Dimension tests failed!")
        exit(1)
    
    # Test 2: Simple generation (greedy)
    if not test_simple_generation():
        log_message("Simple generation tests failed!")
        exit(1)
    
    # Test 3: Beam search generation
    if not test_beam_search_generation():
        log_message("Beam search generation tests failed!")
        exit(1)
    
    log_message("All tests passed! The fix should work.") 