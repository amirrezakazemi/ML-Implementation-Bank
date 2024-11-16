import os
import urllib.request
from torch.utils.data import Dataset, DataLoader
import torch

def encode(text, tokenizer):
    token_ids = torch.tensor(tokenizer.encode(text))
    return token_ids

def decode(token_ids, tokenizer):
    if token_ids.dim() > 1:
        token_ids = token_ids.squeeze(0)
    token_ids = token_ids.tolist()
    text = tokenizer.decode(token_ids)
    return text

def get_data():
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
        
    return text_data

class GPTDataset(Dataset):
    def __init__(self, tokenizer, text_data, context_length, stride):
        super().__init__()

        self.input_ids=[]
        self.labels=[]
        tokenized_text = encode(text_data, tokenizer)

        for i in range(0, len(text_data)-context_length, stride):
            
            self.input_ids.append(tokenized_text[i:i+context_length])
            
            self.labels.append(tokenized_text[i+1:i+1+context_length])

        
    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.input_ids)
    
def create_data_loader(tokenizer, context_length, stride, batch_size, shuffle=True, drop_last=True):
    
    text_data = get_data()
    ratio = 0.8
    train_len = int(len(text_data) * ratio)

    train_data, val_data = text_data[:train_len], text_data[train_len:]


    train_dataset = GPTDataset(tokenizer, train_data, context_length, stride)
    val_dataset = GPTDataset(tokenizer, val_data, context_length, stride)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return train_data_loader, val_data_loader
    





