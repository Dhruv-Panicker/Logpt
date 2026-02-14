import torch
from transformers import GPT2Tokenizer


"""
Class that will handle tokenization for LoGPT and will register special tokens 
for the log data and queries.
"""
class Tokenizer: 

    SPECIAL_TOKENS = {
        'log_start': '<|log_start|>',
        'log_end': '<|log_end|>',
        'query_start': '<|query_start|>',
        'query_end': '<|query_end|>',
        'response_start': '<|response_start|>',
        'response_end': '<|response_end|>',
    }

    def __init__(self): 
        # Load from local cache (no network timeout issues)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
        self.register_special_tokens()

    # Register special tokens with the tokenizer
    def register_special_tokens(self):
        special_tokens_list = list(self.SPECIAL_TOKENS.values())
        num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_list})

    
    #encoding function return as pytorch tensors 
    def encode(self, text: str):
        return self.tokenizer.encode(text)
    
    #decoding function to convert token ids back to text
    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    #get vocab size
    def get_vocab_size(self):
        return len(self.tokenizer)
    
    #get padding token id using EOS token as padding token for simplicity
    def get_pad_token_id(self):
        return self.tokenizer.eos_token_id  
    
    #get tokenizer 
    def get_tokenizer(self):
        return self.tokenizer
