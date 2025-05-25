import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

def debug_original_attention():
    """Debug what the original T5 attention returns"""
    print("[DEBUG] Loading model...")
    
    model_name = "Salesforce/codet5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # Get original attention
    original_attention = model.encoder.block[0].layer[0].SelfAttention
    
    print(f"[DEBUG] Original attention type: {type(original_attention)}")
    print(f"[DEBUG] d_model: {original_attention.d_model}")
    print(f"[DEBUG] n_heads: {original_attention.n_heads}")
    print(f"[DEBUG] key_value_proj_dim: {original_attention.key_value_proj_dim}")
    
    # Create test input
    batch_size, seq_len = 1, 5
    d_model = original_attention.d_model
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    
    print(f"[DEBUG] Test input shape: {hidden_states.shape}")
    
    # Create position bias (required for T5)
    num_heads = original_attention.n_heads
    position_bias = torch.randn(batch_size, num_heads, seq_len, seq_len)
    
    print(f"[DEBUG] Position bias shape: {position_bias.shape}")
    
    # Test original attention
    with torch.no_grad():
        try:
            output = original_attention(
                hidden_states=hidden_states,
                position_bias=position_bias
            )
            print(f"[DEBUG] Original attention output type: {type(output)}")
            
            if isinstance(output, tuple):
                print(f"[DEBUG] Tuple length: {len(output)}")
                for i, item in enumerate(output):
                    if item is not None:
                        print(f"[DEBUG] Item {i}: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'no shape'}")
                    else:
                        print(f"[DEBUG] Item {i}: None")
            else:
                print(f"[DEBUG] Single output shape: {output.shape}")
                
        except Exception as e:
            print(f"[DEBUG] Error: {e}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_original_attention() 