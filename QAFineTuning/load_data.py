from datasets import load_from_disk, load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DefaultDataCollator
import torch

def load_split_data_alpaca(dataset_path):
    dataset = load_from_disk(dataset_path)['train']
    splitted_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    return splitted_dataset['train'], splitted_dataset['test']

def load_split_data_metamathqa(dataset_path):

    dataset = load_dataset("json", data_files=dataset_path)['train']
    splitted_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    return splitted_dataset['train'], splitted_dataset['test']


class AlpacaIFTDataset(Dataset):
    def __init__(self, dataset):

        prompt_without_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request"
        prompt_with_input = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
                
        def process_examples(example):
            
            instructions = [
                f"{prompt_without_input}\n\n### Instruction: {instruction}\n\n ### Response:" if input == ""
                else f"{prompt_with_input}\n\n### Instruction: {instruction}\n\n ### Input: {input}\n\n ### Response:"
                for instruction, input in zip(example["instruction"], example["input"])
            ]
            example["instruction"] = instructions

            return example

        self.dataset = dataset.map(process_examples, batched=True)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        out = {
            'instruction': self.dataset[idx]['instruction'],
            'output': self.dataset[idx]['output'],
        }

        return out


class AlpacaIFTDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    

    def tokenize(self, texts, padding=True):
        
        tokenized_texts = self.tokenizer(
            texts, 
            max_length = 512,
            return_tensors='pt',
            truncation=True,
            padding=padding
        )

        return tokenized_texts

    def __call__(self, batch):

        texts = [
            example['instruction'] +  example['output']
            for example in batch
        ]


        encodings = self.tokenize(texts, padding=True)
        labels = encodings['input_ids'].clone()
        num_labels = 0
        for i, example in enumerate(batch):
            instruction_len = len(self.tokenize(example['instruction'], padding=False).input_ids[0])
            labels[i, :instruction_len] = -100
           
        encodings['labels'] = labels

        return encodings
        
        


class MetaMathQAIFTDataset(Dataset):
    def __init__(self, dataset):

        prompt = "For the following question, write a correct response with explaining your reasoning steps. Let's think step by step." 
                
        def process_examples(example):
            
            questions = [
                f"{prompt}\n\n### Question: {question}\n\n ## Response:" 
                for question in example["query"]
            ]
            example["questions"] = questions
            return example

        self.dataset = dataset.map(process_examples, batched=True)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        out = {
            'question': self.dataset[idx]['questions'],
            'answer': self.dataset[idx]['response'],
        }

        return out
    

class MetaMathQAIFTDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def tokenize(self, texts, padding=True):
        
        tokenized_texts = self.tokenizer(
            texts, 
            max_length = 512,
            return_tensors='pt',
            truncation=True,
            padding=padding
        )

        return tokenized_texts

    def __call__(self, batch):

        texts = [
            example['question'] +  example['answer']
            for example in batch
        ]

        encodings = self.tokenize(texts, padding=True)
        labels = encodings['input_ids'].clone()
        num_labels = 0
        for i, example in enumerate(batch):
            instruction_len = len(self.tokenize(example['question'], padding=False).input_ids[0])
            labels[i, :instruction_len] = -100
           
        encodings['labels'] = labels

        return encodings



# if __name__ == "__main__":
#     model_name = "EleutherAI/gpt-neo-1.3B"  # or "EleutherAI/gpt-neo-2.7B" for larger model
#     dataset_path = "./saved_datasets/alpaca/"

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
#     # Load the Alpaca dataset from Hugging Face
#     dataset = load_dataset("tatsu-lab/alpaca")

#     # Save the dataset locally in JSON format
#     dataset.save_to_disk(dataset_path)

#     train_dataset = IFTDataset(dataset_path)
#     data_collator = IFTDataCollator(tokenizer)
#     train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)

#     print(next(iter(train_dataloader)))