import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import tiktoken
from openai import OpenAI

# Add parent directory to path so we can import from tasks/
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.context import LOG_CONTEXTS, QUERY_TYPES, build_summary_query_prompt, build_varied_query_prompt


# Token limits
MAX_INPUT_TOKENS = 3000  
TARGET_CHUNK_TOKENS = 1000  # Sweet spot for context

# Interface to interact with OpenAI API
class OpenAIClient:
    def __init__(self, api_key: str, encoding_model: str): 
        self.client = OpenAI(api_key=api_key)
        self.enc = tiktoken.get_encoding(encoding_model)

    def chunk_by_tokens(self, lines: List[str], target_tokens: int = TARGET_CHUNK_TOKENS) -> List[str]:
        chunks = []
        current_chunk = [] 
        current_tokens = 0 

        # Iterate through lines and build chunks, more efficient for API calls
        for line in lines: 
            line_tokens = len(self.enc.encode(line))

            # If adding this line exceeds target, save current chunk and start new
            if current_tokens + line_tokens > target_tokens and current_chunk: 
                chunks.append(''.join(current_chunk))
                current_chunk = [] 
                current_tokens = 0 
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add any remaining lines as a final chunk
        if current_chunk: 
            chunks.append(''.join(current_chunk))

        return chunks
    
    #Using Batch API, creating one input file for all log files 
    def create_batch_summary_input(self, all_log_file: Dict[str, str], output_path: str): 
        batch_requests = [] 

        for log_type, log_path_dir in all_log_file.items(): 
            print(f'Processing log type: {log_type}')

            #First read from the log content 
            with open(log_path_dir, "r", encoding="utf-8", errors="ignore") as file:
                lines = file.readlines()

            #Chunk the log lines based on token limits
            chunks = self.chunk_by_tokens(lines, target_tokens=TARGET_CHUNK_TOKENS)
            print(f'Created {len(chunks)} chunks')

            #Create batch requests for each chunk 
            prompt_context = build_summary_query_prompt(log_type)
            
            for chunk_idx, chunk in enumerate(chunks): 
                prompt = f"{prompt_context}\n\nLog Data:\n{chunk}\n\nProvide a concise summary of the above log content."
                request = {
                    "custom_id": f"{log_type}_chunk_{chunk_idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that summarizes log files in a clean clear manner."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 500,
                        "temperature": 0.5
                    }
                }
                batch_requests.append(request)

        #Save batch requests to a single JSONL file
        with open(output_path, "w") as outfile:
            for request in batch_requests: 
                outfile.write(json.dumps(request) + '\n')

        print(f'Batch input file created at {output_path} with {len(batch_requests)} requests.')
        return output_path
    
    #Using Batch API, creating varied input file for multiple query types per log chunk
    def create_varied_batch_input(self, all_log_files: Dict[str, str], output_path: str, query_types: List[str] = None): 
        batch_requests = []
        
        # Use all query types if none specified
        types_to_use = {k: QUERY_TYPES[k] for k in (query_types or QUERY_TYPES.keys())}

        for log_type, log_path_dir in all_log_files.items(): 
            print(f'Processing log type: {log_type}')

            #First read from the log content 
            with open(log_path_dir, "r", encoding="utf-8", errors="ignore") as file:
                lines = file.readlines()

            #Chunk the log lines based on token limits
            chunks = self.chunk_by_tokens(lines, target_tokens=TARGET_CHUNK_TOKENS)
            print(f'Created {len(chunks)} chunks x {len(types_to_use)} query types')

            #Create batch requests for each chunk and each query type
            for chunk_idx, chunk in enumerate(chunks): 
                for query_name, query_info in types_to_use.items(): 
                    #Build prompt for this log type and query
                    prompt_context = build_varied_query_prompt(log_type, query_name)
                    prompt = f"{prompt_context}\n\nLog Data:\n{chunk}"
                    request = {
                        "custom_id": f"{log_type}_chunk_{chunk_idx}_{query_name}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "messages": [
                                {"role": "system", "content": query_info['system']},
                                {"role": "user", "content": prompt}
                            ],
                            "max_tokens": 500,
                            "temperature": 0.5
                        }
                    }
                    batch_requests.append(request)

        #Save batch requests to a single JSONL file
        with open(output_path, "w") as outfile:
            for request in batch_requests: 
                outfile.write(json.dumps(request) + '\n')

        print(f'Varied batch input file created at {output_path} with {len(batch_requests)} requests.')
        return output_path
    
    #Submit the batch request file 
    def submit_batch(self, batch_file_path: str): 
        print(f'Submitting batch file: {batch_file_path}')

        #Upload the file 
        file = self.client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
        )

        #create the batch 
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print(f'Batch created with ID: {batch.id}')

        return batch.id
    
    #Check the batch status 
    def check_batch_status(self, batch_id: str) -> str: 
        batch = self.client.batches.retrieve(batch_id)
        return batch.status
    
    #Get output file ID from batch
    def get_output_file_id(self, batch_id: str) -> str:
        batch = self.client.batches.retrieve(batch_id)
        return batch.output_file_id
    
    #Retrieve batch results
    def retrieve_batch_results(self, output_file_id: str, save_path: str):
        #Downloading results 
        file_response = self.client.files.content(output_file_id)

        #Save to local file
        with open(save_path, "w") as f: 
            f.write(file_response.text)

        print(f'Batch results saved to {save_path}')
        return save_path



        
    

    


    

    

