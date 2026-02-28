import os
import torch
from huggingface_hub import hf_hub_download

from logpt.gpt import GPT
from logpt.tokenizer import Tokenizer
from cli.processor import LogProcessor


HF_REPO_ID = "Dhruv-Panicker/logpt-gpt2-log-analyzer"
HF_MODEL_FILENAME = "best_model.pth"

QUERY_PROMPTS = {
    "summary": "Provide a concise summary of these logs, highlighting key events, errors, and patterns.",
    "root_cause": "Analyze these logs and identify the root cause of any errors or failures.",
    "action_items": "Based on these logs, list specific action items to resolve issues and prevent recurrence.",
}

PROMPT_TEMPLATE = (
    "{log_start}\n{log_content}\n{log_end}\n"
    "{query_start}\n{query}\n{query_end}\n"
    "{response_start}\n"
)

# Returns local checkpoint path, downloading from HuggingFace Hub if not found locally.
def get_model_path() -> str:
    local_candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'logs', 'best_model.pth'),
        os.path.join(os.getcwd(), 'logs', 'best_model.pth'),
    ]
    #check local path
    for path in local_candidates:
        resolved = os.path.abspath(path)
        if os.path.exists(resolved):
            return resolved
    # download from HF
    print(f"Downloading model from {HF_REPO_ID}...")
    return hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)


class LogAnalyzer:

    def __init__(self, model_path: str = None):
        self.device = self._detect_device()
        self.tokenizer = Tokenizer()
        self.special_tokens = self.tokenizer.SPECIAL_TOKENS
        self.response_end_id = self.tokenizer.get_tokenizer().convert_tokens_to_ids(
            self.special_tokens['response_end']
        )

        path = model_path or get_model_path()
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        config = checkpoint.get("config")
        self.model = GPT(config)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.max_seq_len = config.block_size

        self.processor = LogProcessor(
            max_seq_len=self.max_seq_len,
            response_reserve=256
        )
    # Device detection with MPS support for macOS
    def _detect_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # Build the formatted prompt and encode it to token IDs, truncating from the left if needed.
    def _prepare_tokens(self, log_content: str, query_type: str, max_new_tokens: int):
        query = QUERY_PROMPTS[query_type]
        prompt = PROMPT_TEMPLATE.format(
            log_start=self.special_tokens['log_start'],
            log_content=log_content,
            log_end=self.special_tokens['log_end'],
            query_start=self.special_tokens['query_start'],
            query=query,
            query_end=self.special_tokens['query_end'],
            response_start=self.special_tokens['response_start']
        )
        # Tokenize and truncate if total length exceeds model context window (max_seq_len - max_new_tokens)
        tokens = self.tokenizer.encode(prompt)
        max_prompt_len = self.max_seq_len - max_new_tokens
        if len(tokens) > max_prompt_len:
            tokens = tokens[-max_prompt_len:]
        return tokens

    # Streaming autoregressive generation — yields partial decoded text after each new token.
    def generate_stream(self, log_content: str, query_type: str, max_new_tokens: int = 256, temperature: float = 0.7, top_k: int = 50):
        tokens = self._prepare_tokens(log_content, query_type, max_new_tokens)
        # convert to tensor and move to device
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        eos_token_id = self.tokenizer.get_tokenizer().eos_token_id
        generated = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # If input exceeds max_seq_len, condition on the most recent tokens (right truncation)
                x_cond = x if x.size(1) <= self.max_seq_len else x[:, -self.max_seq_len:]
                logits, _ = self.model(x_cond)
                logits = logits[:, -1, :] / temperature
                #get top-k logits and filter
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                # sample from the filtered distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()

                if token_id in (self.response_end_id, eos_token_id):
                    break

                generated.append(token_id)
                x = torch.cat((x, next_token), dim=1)
                yield self.tokenizer.decode(generated).strip()

    # Non-streaming generation — consumes the stream and returns the final result.
    def generate(self, log_content: str, query_type: str, max_new_tokens: int = 256, temperature: float = 0.7, top_k: int = 50):
        result = ""
        for result in self.generate_stream(log_content, query_type, max_new_tokens, temperature, top_k):
            pass
        return result

    # Analyze a single chunk of the log file and return the result with chunk metadata.
    def analyze(self, log_text: str, task: str, chunk_index: int = 0) -> dict:
        chunks = self.processor.chunk(log_text)
        total = len(chunks)

        # if chunk index out of range then return empty response with done=True
        if chunk_index >= total:
            return {"response": "", "chunk": chunk_index, "total_chunks": total, "done": True}

        response = self.generate(chunks[chunk_index], task)
        return {
            "response": response,
            "chunk": chunk_index + 1,
            "total_chunks": total,
            "done": chunk_index + 1 >= total,
        }
    # function that Analyze all chunks of the log file sequentially.
    def get_all_chunks(self, log_text: str, task: str) -> list[dict]:
        chunks = self.processor.chunk(log_text)
        results = []

        for i in range(len(chunks)):
            response = self.generate(chunks[i], task)
            results.append({
                "response": response,
                "chunk": i + 1,
                "total_chunks": len(chunks),
                "done": i + 1 >= len(chunks),
            })

        return results
