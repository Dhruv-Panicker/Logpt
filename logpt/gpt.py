from dataclasses import dataclass
import torch 
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

'Self Attention Class Implementation'
class CausalSelfAttention(nn.Module): 

    def __init__(self, config): 
        super().__init__()

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batched way
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # do I need to add dropput if I am doing regularization already?

    #in GPT-2 (124M), n_head=12, hs=64, so nh(numofheads)*hs(headsize)=C=768 channels in the Transformer
    def forward(self, x):
        #batch size, channels, embedding size
        B, T, C = x.size()
        # compute attention scores ("affinities") 
        # calculate the query, key, value for all heads in batch and move head forward to be the batch dimension 
        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # view new tensor as (B, num of heads, T, head size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, channels, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, channels, hs)
        # implement flash attention, and evaluate how much does each token attend to another 
        #Computes scaled dot product attention on query, key and value 
        # apply softmax to noarmalize the attention scores so that they add up to 1
        # get weighted average of the values
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
'Multi-Level Perceptron Class Implementation'

class MLP(nn.Module): 

    def __init__(self, config): 
        super().__init__()
        # fully connected layer with 4x dimension        
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU to allow for smaller negative values and smoother curve
        self.c_gelu = nn.GELU(approximate='tanh')
        # project back to original embedding size
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_gelu(x)
        x = self.c_proj(x)
        return x
    

'Transformer Block Class Implementation'

class Block(nn.Module):
    def __init__(self, config): 
        super().__init__()
        # layer norm 1
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # self attention
        self.attn = CausalSelfAttention(config)
        # layer norm 2
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # MLP
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) #map through layer norm and attention then add residual
        x = x + self.mlp(self.ln_2(x))  #map through layer norm and MLP then add residual
        return x
    

'GPT Model Confiuration Dataclass folllowing GPT-2 (124M) Architecture'
@dataclass
class GPTConfig:
    vocab_size: int = 50257  # number of tokens: 50,000 from BPE + 256 bytes tokens + 1 speacial <EOS> token
    block_size: int = 1024  # max sequence length
    n_embd: int = 768        # embedding dimension
    n_head: int = 12         # number of attention heads
    n_layer: int = 12        # number of layers 


'GPT Model Implementation'
class GPT(nn.Module): 

    def __init__(self, config): 
        super().__init__()
        self.config = config 

        # Create transformer collection 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer blocks
            ln_f = nn.LayerNorm(config.n_embd) # final layer norm
        ))

        #final projection layer output layer 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size() # batch size, sequence length
        assert T <= self.config.block_size, "Cannot forward, the seequence length cannot exceed the block size"

        #forward the token embedding and position embedding
        tok_embd = self.transformer.wte(idx) # shape (B, T, C or n_embd)

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_embd = self.transformer.wpe(pos) # shape (T, n_embd)

        x = tok_embd + pos_embd # shape (B, T, C or n_embd)
        #forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # forward through final layer norm
        x = self.transformer.ln_f(x) # shape (B, T, C or n_embd)
        logits = self.lm_head(x) # shape (B, T, vocab_size) get the logits (probabilities for each token in the vocabulary)
        return logits 
    
    
    """Loads pretrained GPT-2 model weights from huggingface"""
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], "model_type must be one of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'"
        print(f'loading weights from pretrained {model_type} model...')

        #get n_layer and n_embd based on model type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        #initialize GPT model with config
        config = GPTConfig(**config_args)
        model = GPT(config)

        #Get the dictionary of the pretrained GPT-2 model from huggingface
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard the this mask / buffer parameters from state dict keys
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.masked_bias')]

        #initialize huggingface GPT-2 model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        #transpose the weights of the Conv1D layers
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model





    










