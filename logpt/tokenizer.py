import tiktoken
import torch

def get_tokenizer(model_name: str, num_return_sequences: int = 1):
    enc = tiktoken.get_encoding(model_name)
    tokens = enc.encode('Hello, I am a language model.')
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # Add batch dimension and repeat for num_return_sequences
    x = tokens.to('cuda' if torch.cuda.is_available() else 'cpu')
    return enc, x