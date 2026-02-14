from logpt.tokenizer import Tokenizer


'Split a log file into model-sized chunks' 
class LogProcessor: 
    def __init__(self, max_seq_len: int = 1024, response_reserve: int = 256):
        self.max_seq_len = max_seq_len
        self.response_reserve = response_reserve
        self.tokenizer = Tokenizer()
        # prompt template tokens 
        self.prompt_overhead = 80 

        self.num_chunks = max_seq_len - response_reserve - self.prompt_overhead

    #Function that will split log content chunks that fit into the model's context window 
    def chunk(self, log_content: str):
        if not log_content.strip():
            return ["(empty log content)"]
        lines = log_content.splitlines()
        chunks = []
        current_line = [] 
        current_tokens = 0 

        for line in lines: 
            line_tokens = len(self.tokenizer.encode(line))

            #if adding this current chunk exceeds model limit then finalize current chunk and start a new one
            if current_tokens + line_tokens > self.num_chunks and current_line:
                chunks.append("\n".join(current_line))
                current_line = []
                current_tokens = 0
            #if a single line exceeds the limit, truncate it 
            if line_tokens > self.num_chunks: 
                tokens = self.tokenizer.encode(line)[:self.num_chunks]
                line = self.tokenizer.decode(tokens)
                line_tokens = self.num_chunks
            
            current_line.append(line)
            current_tokens += line_tokens

        #add any remaining lines as a final chunk
        if current_line:
            chunks.append("\n".join(current_line))
        return chunks



