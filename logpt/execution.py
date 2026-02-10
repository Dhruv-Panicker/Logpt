
"""
LoGPT Training Loop 
Fine-tuning GPT-3 on log analysis using MPS/CUDA/CPU
"""

import os
import json
import math
import time
import torch
import torch.nn.functional as F
from logpt.tokenizer import Tokenizer
from logpt.dataloader import Dataloader
from logpt.gpt import GPT, GPTConfig

#Batch configurations 
B = 2 #micro batch size for MPS
T = 1024 #sequence length for GPT-2
total_batch_size = 16 #32 # desired batch size
assert total_batch_size % B == 0, "Total batch size must be divisible by micro batch size B"
# using gradient accumulation to achieve effective batch size of 32 while only processing 4 samples at a time on MPS/CPU
grad_accum_steps = total_batch_size // B


#Learning rate configurations 
max_lr = 2e-5
min_lr = max_lr * 0.1
warmup_steps = 100

max_steps = 2000

#Data configuration 
data_path = "data/training/training_data.jsonl"
train_ratio = 0.85
val_ratio = 0.10
seed = 42

#Weight decay 
weight_decay = 0.1

#Early stopping 
eval_interval = 50  # evaluate every N steps
patience = 10        # stop after N evals with no improvement

#Function that will implement cosine learing rate with linear warmup and then cosine decay after that
def get_lr(step): 
    #1) linear warmup for the first warmup_steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    #2) if steps > learning_rate_decay_iters then return min_lr
    if step > max_steps: 
        return min_lr
    #3) cosine decay from max_lr to min_lr between warmup_steps and learning_rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #cosine coeff
    return min_lr + coeff * (max_lr - min_lr)

#Function that will load the records from the training file 
def load_records(data_path): 
    records = [] 
    with open(data_path, 'r') as f: 
        for line in f: 
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {data_path}")
    return records

# Configure AdamW optmimizer with weight decay applied on weight tensors but not on bias or layernorm parameters
def configure_optimizer(model, lr, weight_decay): 

    # Get those parameters that should and should not have weight decay
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} #only consider parameters that require grad

    #Sepreate out those that require weight decay and those that don't
    decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
    non_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': non_decay_params, 'weight_decay': 0.0}
    ]

    #count the number of decay and non decay params 
    num_decay_params = sum(p.numel() for p in decay_params)
    num_non_decay_params = sum(p.numel() for p in non_decay_params)
    print(f"Optimizer configured with {num_decay_params} decay params and {num_non_decay_params} non-decay params")

    #optimize 
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    return optimizer


def main(): 
    #Device setup 
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    torch.manual_seed(seed)

    #Tokenizer 
    print("Loading tokenizer...")
    tokenizer = Tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer loaded with vocab size: {vocab_size}")

    #Load Data 
    print("Loading records...")
    records = load_records(data_path)

    #Dataloader
    print("Initializing dataloader...")
    train_loader = Dataloader(B, T, records, tokenizer, split='train', data_path=data_path, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    val_loader = Dataloader(B, T, records, tokenizer, split='val', data_path=data_path, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    #Model 
    print("Loading pretrained GPT-2 model...")
    model = GPT.from_pretrained("gpt2")
    
    # Resize embeddings for special tokens
    model.resize_token_embeddings(vocab_size)
    print("Model loaded and resized for special tokens.")

    model.to(device)
    optimizer = configure_optimizer(model, max_lr, weight_decay)

    #Add logging 
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"training_log_{int(time.time())}.txt")
    #write batch size 
    with open(log_path, 'w') as f: 
        f.write("step,train_loss,val_loss,lr\n")
        f.write(f"Max LR: {max_lr}, Min LR: {min_lr}\n")
        f.write(f"Effective batch size: {total_batch_size}\n")
        f.write(f"Grad accumulation steps: {grad_accum_steps}\n")
        f.write(f"Warmup steps: {warmup_steps}, Max steps: {max_steps}\n")
        f.write(f"# Weight decay: {weight_decay}\n")
    
    #training loop
    print(f"Starting training for {max_steps} steps (early stopping patience={patience})...")
    best_val_loss = float('inf')
    patience_counter = 0

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        #evaluate the model on val set every eval_interval steps
        if (step > 0 and step % eval_interval == 0) or last_step: 
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = min(100, len(val_loader))
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.item()

            print(f"Step {step}: Val loss: {val_loss_accum:.4f}")

            #Log validation 
            with open(log_path, 'a') as f: 
                f.write(f"{step},, {val_loss_accum:.4f}, {get_lr(step):.6e}\n")
            
            #Save the best model and check early stopping
            if val_loss_accum < best_val_loss:
                best_val_loss = val_loss_accum
                patience_counter = 0  
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': best_val_loss
                }
                torch.save(checkpoint, os.path.join(log_dir, "best_model.pth"))
                print(f"New best model saved with val loss {best_val_loss:.4f} at step {step}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter}/{patience} evals (best: {best_val_loss:.4f})")
                if patience_counter >= patience:
                    print(f"Early stopping triggered at step {step}! No improvement for {patience} evals.")
                    with open(log_path, 'a') as f:
                        f.write(f"EARLY STOP at step {step}, best val loss: {best_val_loss:.4f}\n")
                    break
        
        #training 
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps): 
            x, y, = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)
            loss = loss / grad_accum_steps #normalize loss for gradient accumulation
            loss_accum += loss.item()
            #backpropagate
            loss.backward()
        
        #Calculate the global norm of the parameters, preven the model from gettting big shocks with large gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        #determine the learning rate and set the learning rate for this iteration 
        lr = get_lr(step)
        #manually set the learning rate for each param group in the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = grad_accum_steps * B * T
        tokens_per_sec = tokens_processed / dt
        print(f"Step {step}: Train loss: {loss_accum:.4f}, LR: {lr:.6e}, Time: {dt:.2f}s, Tokens/sec: {tokens_per_sec:.2f}, Grad norm: {norm:.4f}")

        #Logging simple 
        with open(log_path, 'a') as f: 
            f.write(f"{step}, {loss_accum:.4f}, , {lr:.6e}\n")
        
    #save final model at the end of training
    checkpoint = {
        'model': model.state_dict(),
        'config': model.config,
        'step': max_steps,
        'val_loss': best_val_loss
    }
    torch.save(checkpoint, os.path.join(log_dir, f"final_model.pth"))
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
