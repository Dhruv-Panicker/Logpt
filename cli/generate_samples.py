import sys
import torch
import tiktoken

from logpt.gpt import GPT, GPTConfig
from logpt.tokenizer import Tokenizer


#Three types of query types 
QUERY_PROMPTS = {
    "summary": "Provide a concise summary of these logs, highlighting key events, errors, and patterns.",
    "root_cause": "Analyze these logs and identify the root cause of any errors or failures.",
    "action_items": "Based on these logs, list specific action items to resolve issues and prevent recurrence.",
}

'Load lastest checkpoint and generate the log analysis' 
class LogAnalyzer: 

    PROMPT_TEMPLATE = (
        "{log_start}\n{log_content}\n{log_end}\n"
        "{query_start}\n{query}\n{query_end}\n"
        "{response_start}\n"
    )

    def __init__(self, checkpoint_path, device=None): 
        self.device = device or self._detect_device()
        self.tokenizer = Tokenizer()
        self.special_tokens = self.tokenizer.SPECIAL_TOKENS
        self.response_end_id = self.tokenizer.get_tokenizer().convert_tokens_to_ids(self.special_tokens['response_end'])

        #load the latest model checkpoint 
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        config = checkpoint.get("config")

        self.model = GPT(config)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.max_seq_len = config.block_size

        print(f"Model loaded successfully on device: {self.device}")
    
    def _detect_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    #Function to generate response from the model given log content and query type
    @torch.no_grad()
    def generate(self, log_content, query_type, max_new_tokens=256, temperature=0.7, top_k=50): 
        query = QUERY_PROMPTS[query_type]

        prompt = self.PROMPT_TEMPLATE.format(
            log_start=self.special_tokens['log_start'],
            log_content=log_content,
            log_end=self.special_tokens['log_end'],
            query_start=self.special_tokens['query_start'],
            query=query,
            query_end=self.special_tokens['query_end'],
            response_start=self.special_tokens['response_start']
        )

        tokens = self.tokenizer.encode(prompt)

        #Ensure that the prompt fits 
        max_prompt_len = self.max_seq_len - max_new_tokens
        if len(tokens) > max_prompt_len: 
            tokens = tokens[-max_prompt_len:]

        x = torch.tensor([tokens], dtype=torch.long, device=self.device)

        generated = [] 
        eos_token_id = self.tokenizer.get_tokenizer().eos_token_id
        for _ in range(max_new_tokens): 
            x_cond = x if x.size(1) <= self.max_seq_len else x[:, -self.max_seq_len:]
            logits, _ = self.model(x_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()

            #stop conditions 
            if token_id == self.response_end_id: 
                break 
            if token_id == eos_token_id: 
                break 
            generated.append(token_id)
            x = torch.cat((x, next_token), dim=1)
        response = self.tokenizer.decode(generated)
        return response.strip()

                