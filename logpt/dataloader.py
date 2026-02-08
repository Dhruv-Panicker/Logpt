from collections import defaultdict
import json
import os
import numpy as np
import torch
import tiktoken
from pathlib import Path

"""
Class that will load training data and create batches for training the model. 
"""

class Dataloader: 

    def __init__(self, B: int, T: int, records, tokenizer, split, data_path, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.15): 
        self.B = B # batch size
        self.T = T # sequence length 1024 for GPT-2
        self.tokenizer = tokenizer

        assert split in {'train', 'val'}, f'split myst be either train or val'
        self.split = split
        
        #Load and process the data 
        split_records = self._shuffle_and_split(records, seed, train_ratio, val_ratio)
        self.records = split_records[split]

        #Tokenize into flat tensor 
        self.tokens = self._tokenize_records(self.records)
        self.current_position = 0

    #Function that will shuffle the records and make train/val split
    def _shuffle_and_split(self, records, seed, train_ratio, val_ratio):
        rng = np.random.RandomState(seed)

        #Group by (query_type, log_type) for stratified split
        groups = defaultdict(list)
        for record in records: 
            key = (record.get('query_type'), record['log_type'])
            groups[key].append(record)

            train_records = [] 
            val_records = [] 

        for key, group_records in groups.items():
            rng.shuffle(group_records)
            n = len(group_records)
            n_train = int(train_ratio * n)
            n_val = int(val_ratio * n)
            #train records up to n_train val then rest val records
            train_records.extend(group_records[:n_train])
            val_records.extend(group_records[n_train:n_train+n_val])

        #final shuffle again 
        rng.shuffle(train_records)
        rng.shuffle(val_records)

        return {
            'train': train_records,
            'val': val_records
        }
    
    #Function that will tokenize the records into a flat tensor for training
    def _tokenize_records(self, records):
        all_token_ids = [] 
        padding_token_id = self.tokenizer.get_pad_token_id()

        for i, record in enumerate(records): 
            text = record['text']
            #this retunrs a tensor of shape (1, seq_len)
            token_ids = self.tokenizer.encode(text)

            if len(token_ids) > self.T: 
                token_ids = token_ids[:self.T]
            elif len(token_ids) < self.T:
                #pad with padding token id to reach sequence length T
                token_ids = token_ids + [padding_token_id] * (self.T - len(token_ids))
            
            all_token_ids.extend(token_ids)

            if (i + 1) % 1000 == 0:
                print(f"  Tokenized {i + 1}/{len(records)} records")

        return torch.tensor(all_token_ids, dtype=torch.long)
    
    #Function to reset position for new epoch
    def reset(self):
        self.current_position = 0

    #Function to get the next batch 
    def next_batch(self): 
        B, T = self.B, self.T
        end_pos = self.current_position + B * T + 1 # +1 for target shift

        #If end exceeds wrap around and start new epoch
        if end_pos > len(self.tokens):
            self.reset()
            end_pos = self.current_position + B * T + 1
        
        buffer = self.tokens[self.current_position:end_pos]
        x = buffer[:-1].view(B, T) # input tokens
        y = buffer[1:].view(B, T)  # target tokens (shifted by 1)

        self.current_position += B * T

        return x, y
    
    #Function to get length of the dataset in terms of number of batches
    def __len__(self):
        return len(self.tokens) // (self.B * self.T)




    