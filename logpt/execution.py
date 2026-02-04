import torch.nn.functional as F
import torch
import tiktoken
from logpt.tokenizer import get_tokenizer
from logpt.gpt import GPT, GPTConfig



num_return_sequences = 5
max_length = 30

torch.manual_seed(42)

'Function to sample new tokens from the model given a prompt'
def generate_samples(model, x, enc, max_length, num_return_sequences):
    while x.size(1) < max_length:
        #forward the model to get all the logits 
        logits = model(x) # (B, T, C)
        #only care about the logit at the last column 
        logits = logits[:, -1, :] # (B, vocab_size)
        #Apply softmax to get porbabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling from huggingface
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        # meaning we only consider the top 50 most likely next tokens
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
        # sample from the top-k probs
        ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
        #gather the corresponding token indices
        xcol = torch.gather(topk_indices, -1, ix) #(B, 1)
        #append to the sequence and continue
        x = torch.cat((x, xcol), dim=1) 


    #print the generated text 
    for i in range(num_return_sequences): 
        tokens = x[i, :max_length].tolist()
        text = enc.decode(tokens)
        print("> ", text)

#execute 
if __name__ == "__main__":
    model_name = 'gpt2'
    enc, x = get_tokenizer(model_name, num_return_sequences)

    #load pretrained gpt2 model
    model = GPT.from_pretrained(model_name)
    print("Model loaded successfully.")
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    #generate samples
    generate_samples(model, x, enc, max_length, num_return_sequences)
