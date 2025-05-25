import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

def test_base_model_generation():
    """Test base model generation with various prompts"""
    print("=== TESTING BASE MODEL GENERATION ===")
    
    # Initialize model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # Test prompts
    test_prompts = [
        "def add_numbers(a, b):",
        "# Function to add two numbers\ndef add(",
        "Generate a Python function that adds two numbers",
        "Write Python code: def add_two_numbers",
        "Python function: add two integers",
        "def calculate_sum(x, y):",
        "# Add two numbers\ndef",
    ]
    
    model.eval()
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        # Tokenize
        input_encoding = tokenizer(
            prompt, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        try:
            with torch.no_grad():
                # Generate with different parameters
                outputs = model.generate(
                    input_ids=input_encoding.input_ids,
                    attention_mask=input_encoding.attention_mask,
                    max_length=100,
                    num_beams=1,  # Greedy
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Generated: {generated_text}")
                print(f"Length: {len(generated_text)} characters")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n=== BASE MODEL TEST COMPLETE ===")

if __name__ == "__main__":
    test_base_model_generation() 