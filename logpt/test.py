"""
Test Logpt model on held-out test data 
Evaluate model performance and genreates predictions 
"""
import os
import json
import torch
import math
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
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
        generated = generate_response(model, tokenizer, prompt, device, sequence_length)

        #Compute rouge scores
        scores = compute_rouge_scores(ground_truth, generated)

        #Compute BERTScore
        bert_scores = compute_bertscore([generated], [ground_truth], lang='en')

        #print results to console
        print(f"Prompt:\n{prompt}\n")
        print("===" * 20)
        print(f"Ground Truth:\n{ground_truth}\n")
        print("===" * 20)
        print(f"Generated Response:\n{generated}\n")
        print("-" * 50)

        if scores:
            print(f"ROUGE-1 F1: {scores['rouge1_f']:.4f} | ROUGE-2 F1: {scores['rouge2_f']:.4f} | ROUGE-L F1: {scores['rougeL_f']:.4f}")
        print("-" * 50)

        if bert_scores:
            print(f"BERTScore Precision: {bert_scores['precision'][0]:.4f} | Recall: {bert_scores['recall'][0]:.4f} | F1: {bert_scores['f1'][0]:.4f}")
        print("-" * 50)


        results.append({
            'prompt': prompt,
            'ground_truth': ground_truth,
            'generated': generated,
            'query_type': record.get('query_type', 'unknown'),
            'log_type': record.get('log_type', 'unknown'),
            'rouge_scores': scores,
            'bert_scores': bert_scores
        })
    
    return results
    
# Function to generate a response given a prompt
def generate_response(model, tokenizer, prompt, device, sequence_length, max_token=500, temperature=0.7):
    model.eval()

    tokens = tokenizer.encode(prompt)
    prompt_length = len(tokens)
    #unsqueeze to dimensions (1, seq_len) for batch size of 1
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    generated = tokens.copy()

    with torch.no_grad():
        for _ in range(max_token):
            #only feed the last sequence_length tokens to the model to stay within context window
            input_tokens = generated[-sequence_length:]
            x = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)

            logits, _ = model(x, None)
            logits = logits[0, -1, :] / temperature

            # sample next token 
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

            #stop at response_end or pad token 
            if next_token == tokenizer.get_pad_token_id():
                break

            decoded_so_far = tokenizer.decode(generated)
            if '<|response_end|>' in decoded_so_far:
                break

    #Only decode the NEW tokens (after the prompt)
    generated_tokens = generated[prompt_length:]
    response_text = tokenizer.decode(generated_tokens)

    #clean up remove <|response_end|> if present and anything after it 
    response_end = '<|response_end|>'
    end_idx = response_text.find(response_end)
    if end_idx != -1:
        response_text = response_text[:end_idx].strip()
    return response_text.strip()

#Function that will compute the similarity scores using ROUGE
#Compute ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence)
def compute_rouge_scores(ground_truth, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, generated)
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge1_p': scores['rouge1'].precision,
        'rouge1_r': scores['rouge1'].recall,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rouge2_p': scores['rouge2'].precision,
        'rouge2_r': scores['rouge2'].recall,
        'rougeL_f': scores['rougeL'].fmeasure,
        'rougeL_p': scores['rougeL'].precision,
        'rougeL_r': scores['rougeL'].recall,
    }

#Function to compute the bERTScore similarity using the bert_score library (if available)
def compute_bertscore(generated_list, ground_truth_list, lang='en', batch_size=5): 
    # Auto-detect best device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Computing BERTScore on device: {device}")
    #add some progress logging for long computations
    total = len(generated_list)
    #total progress 
    [print(f"  BERTScore progress: {i}/{total} ({(i/total)*100:.2f}%)") for i in range(0, total, batch_size)]
    
    P, R, F1 = bert_score(generated_list, ground_truth_list, lang=lang, device=device, batch_size=batch_size)

    return {
        'precision': P.numpy().tolist(), 
        'recall': R.numpy().tolist(),
        'f1': F1.numpy().tolist()
    }



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

    #Load the best model checkpoint (auto-detect latest)
    checkpoint_path = "logs/best_model.pth"
    if not os.path.exists(checkpoint_path):
        # fallback: find any best_model_step_*.pth
        import glob
        candidates = sorted(glob.glob("logs/best_model_step_*.pth"))
        if candidates:
            checkpoint_path = candidates[-1]
        else:
            print("No checkpoint found in logs/!")
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
    perplexity, avg_loss, num_evaluated = eval_model(model, tokenizer, test_records, sequence_length=1024, device=device)
    print(f"Evaluated {num_evaluated} records.")
    print(f"Test Set Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

    #Sample generation
    print("Generating sample responses...")
    results = test_sample_generation(model, tokenizer, test_records, device, sequence_length=1024, num_samples=5)

    # Print aggregate ROUGE scores
    if results and results[0].get('rouge_scores'):
        avg_rouge1 = np.mean([r['rouge_scores']['rouge1_f'] for r in results if r.get('rouge_scores')])
        avg_rouge2 = np.mean([r['rouge_scores']['rouge2_f'] for r in results if r.get('rouge_scores')])
        avg_rougeL = np.mean([r['rouge_scores']['rougeL_f'] for r in results if r.get('rouge_scores')])
        print(f"\n{'=' * 60}")
        print(f"Aggregate ROUGE Scores ({len(results)} samples):")
        print(f"  ROUGE-1 F1: {avg_rouge1:.4f}")
        print(f"  ROUGE-2 F1: {avg_rouge2:.4f}")
        print(f"  ROUGE-L F1: {avg_rougeL:.4f}")
        print(f"{'=' * 60}")
    
    # Print aggregate BERTScore 
    if results and results[0].get('bert_scores'):
        avg_bert_precision = np.mean([r['bert_scores']['precision'][0] for r in results if r.get('bert_scores')])
        avg_bert_recall = np.mean([r['bert_scores']['recall'][0] for r in results if r.get('bert_scores')])
        avg_bert_f1 = np.mean([r['bert_scores']['f1'][0] for r in results if r.get('bert_scores')])
        print(f"\n{'=' * 60}")
        print(f"Aggregate BERTScore ({len(results)} samples):")
        print(f"  Precision: {avg_bert_precision:.4f}")
        print(f"  Recall: {avg_bert_recall:.4f}")
        print(f"  F1: {avg_bert_f1:.4f}")
        print(f"{'=' * 60}")
    
    # Save results to JSON for notebook analysis
    results_path = os.path.join("logs", "test_results.json")
    serializable_results = []
    for r in results:
        serializable_results.append({
            'query_type': r['query_type'],
            'log_type': r['log_type'],
            'ground_truth': r['ground_truth'],
            'generated': r['generated'],
            'rouge_scores': r['rouge_scores'],
            'bert_scores': r['bert_scores'],
        })
    with open(results_path, 'w') as f:
        json.dump({
            'avg_loss': avg_loss,
            'perplexity': perplexity,
            'results': serializable_results,
        }, f, indent=2)
    print(f"Results saved to {results_path}")

    print("Sample generation complete.")


if __name__ == "__main__":
    main()








         

