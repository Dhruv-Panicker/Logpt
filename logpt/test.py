"""
Test Logpt model on held-out test data 
Evaluate model performance and genreates predictions 
"""
import os
import json
import torch
import math
import numpy as np
from logpt.tokenizer import Tokenizer
from logpt.gpt import GPT, GPTConfig

def load_test_data(data_path, test_ratio=0.05, seed=42):
    records = [] 
    with open(data_path, 'r') as f: 
        for line in f: 
            records.append(json.loads(line))
    np.random.seed(seed)
    np.random.shuffle(records)

    train_end = int(len(records) * 0.85)
    val_end = int(len(records) * 0.95)
    test_records = records[val_end:]

    print(f"Loaded {len(test_records)} test records")
    return test_records

# Function to calculate the perplexity of the model on the test set
def eval_model(model, tokenizer, test_records, sequence_length, device): 
    model.eval()
    total_loss = 0.0 
    total_tokens = 0 
    num_evaluated = 0 

    with torch.no_grad(): 
        for i, record in enumerate(test_records): 
            text = record['text']
            tokens = tokenizer.encode(text)

            #if token length too small skip 
            if len(tokens) < 10: 
                continue 
            #truncate if too large for the model
            if len(tokens) > sequence_length:
                tokens = tokens[:sequence_length]

            x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)

            logits, loss = model(x, y)

            total_loss += loss.item() * (len(tokens) - 1)
            total_tokens += len(tokens) - 1 
            num_evaluated += 1

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss, num_evaluated

# Function to generate responses for a sample test cases 
def test_sample_generation(model, tokenizer, test_records, device, sequence_length, num_samples=5):
    
    model.eval()
    results = [] 

    #select diverse samples 
    selected = np.random.choice(len(test_records), min(num_samples, len(test_records)), replace=False)

    for idx in selected: 
        record = test_records[idx]
        #Have to extract log and query from the formatted text
        text = record['text']

        # Find the prompt (everything up to <|response_start|>)
        response_start_token = '<|response_start|>'
        response_end_token = '<|response_end|>'

        if response_start_token not in text: 
            continue 

        prompt = text[:text.find(response_start_token) + len(response_start_token)]

        #Extract ground truth 
        start_idx = text.find(response_start_token) + len(response_start_token)
        end_idx = text.find(response_end_token)
        if end_idx == -1:
            end_idx = len(text)
        # removing the response to generate and evaluate against the ground truth
        ground_truth = text[start_idx:end_idx].strip()

        #generate prediction 
        generated = generate_response(model, tokenizer, prompt, device, sequence_length, max_token=300)

        #print results to console
        print(f"Prompt:\n{prompt}\n")
        print(f"Generated Response:\n{generated}\n")
        print("-" * 50)


        results.append({
            'prompt': prompt,
            'ground_truth': ground_truth,
            'generated': generated,
            'query_type': record.get('query_type', 'unknown'),
            'log_type': record.get('log_type', 'unknown')
        })
    
        return results
    
# Function to generate a response given a prompt
def generate_response(model, tokenizer, prompt, device, sequence_length, max_token=300, temperature=0.7):
    model.eval()

    tokens = tokenizer.encode(prompt)
    #unsqueeze to dimensions (1, seq_len) for batch size of 1
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    generated = tokens.copy()

    with torch.no_grad():
        for _ in range(max_token):
            logits, _ = model(x, None)
            logits = logits[0, -1, :] / temperature

            # sample next token 
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

            #stop at response_end or pad token 
            if next_token == tokenizer.get_pad_token_id():
                break

            decoded = tokenizer.decode(generated)
            if '<|response_end|>' in decoded:
                break

            #update the input and unsqueeze for dimensions (1, seq_len)
            x = torch.tensor(generated[-sequence_length:], dtype=torch.long, device=device).unsqueeze(0)
    
    full_text = tokenizer.decode(generated)

    #Extract just the generated response 
    response_start = '<|response_start|>'
    response_end = '<|response_end|>'

    start_idx = full_text.find(response_start)
    if start_idx == -1:
        start_idx += len(response_start)
        end_idx = full_text.find(response_end, start_idx)
        if end_idx == -1:
            end_idx = len(full_text)
        return full_text[start_idx:end_idx].strip()
    
    return full_text


def main():
    # Device setup
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"

    print("="*60)
    print("LogPT Model Evaluation")
    print("="*60)

    #Load tokenizer 
    print("Loading tokenizer...")
    tokenizer = Tokenizer()

    #Load the best model checkpoint
    checkpoint_path = "logs/best_model_step_1999.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found!")
        return
    print(f"Loading model from {checkpoint_path}...")
    torch.serialization.add_safe_globals([GPTConfig])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print("Model loaded.")

    #load the test data 
    test_records = load_test_data("data/training/training_data.jsonl")

    #Evaluate model on test set
    print("Evaluating model on test set...")
    perplexity, avg_loss, num_evaluated = eval_model(model, tokenizer, test_records, sequence_length=512, device=device)
    print(f"Evaluated {num_evaluated} records.")
    print(f"Test Set Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

    #Sample generation
    print("Generating sample responses...")
    sample_results = test_sample_generation(model, tokenizer, test_records, device, sequence_length=512, num_samples=5)
    print("Sample generation complete.")


if __name__ == "__main__":
    main()








         

