from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from load_data import AlpacaIFTDataset, AlpacaIFTDataCollator, \
    load_split_data_alpaca, load_split_data_metamathqa, \
    MetaMathQAIFTDataset, MetaMathQAIFTDataCollator
from torch.utils.data import DataLoader, Subset
from evaluate import load  # New way to load metrics
import torch.nn as nn
import torch
import random
import bert_score


model_path = "/home/ubuntu/volume/meta-llama-3-8b-instruct"
dataset_path = "/home/ubuntu/volume/MetaMathQA-395k.jsonl"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# model.resize_token_embeddings(len(tokenizer))

tokenizer.pad_token_id = tokenizer.eos_token_id

# train_dataset, eval_dataset = load_split_data_alpaca(dataset_path)

# train_dataset = AlpacaIFTDataset(train_dataset)

# eval_dataset = AlpacaIFTDataset(eval_dataset)

# data_collator = AlpacaIFTDataCollator(tokenizer)

train_dataset, eval_dataset = load_split_data_metamathqa(dataset_path)

train_dataset = MetaMathQAIFTDataset(train_dataset)

eval_dataset = MetaMathQAIFTDataset(eval_dataset)

data_collator = MetaMathQAIFTDataCollator(tokenizer)

# dataloader = DataLoader(dataset, batch_size=2, collate_fn=data_collator)

# print(next(iter(dataloader)))

# Load the ROUGE metric
rouge = load("rouge")
bertscore = load("bertscore")
accuracy = load("accuracy")

def compute_rougel(preds, refs):
    # Compute ROUGE-L
    rouge_result = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
    rouge_l_score = rouge_result["rougeL"]

    return rouge_l_score

def compute_bert_score(preds, refs):

    # Compute BERTScore
    bertscore_result = bertscore.compute(predictions=preds, references=refs, lang="en")
    bertscore_f1 = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])  # Averaging F1 scores

    return bertscore_f1

def compute_accuracy(preds, refs):
    return accuracy.compute(predictions=preds, references=refs)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    preds = predictions.argmax(-1)

    ignore_labels = [-100, tokenizer.pad_token_id]

    preds_decoded = []
    labels_decoded = []

    preds_acc = []
    labels_acc = []

    for i in range(len(preds)):
        
        mask = ~torch.isin(torch.tensor(labels[i]), torch.tensor(ignore_labels))
        preds_mask = torch.cat((mask[1:], torch.tensor([False])))

        preds_acc += preds[i, preds_mask].tolist()
        labels_acc += labels[i, mask].tolist()

        preds_decoded.append(
            tokenizer.decode(preds[i, preds_mask], skip_special_tokens=True)
        )
        labels_decoded.append(
            tokenizer.decode(labels[i, mask], skip_special_tokens=True)
        )

        print(f'gt label {labels_decoded[-1]}')
        print(f'prediction {preds_decoded[-1]}')

    accuracy = compute_accuracy(preds_acc, labels_acc)['accuracy']
    rougel = compute_rougel(preds_decoded, labels_decoded)
    bert_score = compute_bert_score(preds_decoded, labels_decoded)
    
    print(rougel)
    print(bert_score)
    print(accuracy)

    return {
        "rouge-l": rougel,
        "bert-score": bert_score,
        "accuracy": accuracy,
    }

# 6. Custom Trainer with `compute_loss` Override
class CustomTrainer(Trainer):
    
    def __init__(self, *args, eval_subset_size=10, tokenizer=tokenizer, eval_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_subset_size = eval_subset_size  # Define subset size for evaluation
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

    def nll_loss(self, logits, labels, shift_labels=True):
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
        
        labels = labels.flatten()
        logits = logits.reshape(-1, logits.shape[-1])

        ignore_labels = [-100, self.tokenizer.pad_token_id]
        mask = ~torch.isin(labels, torch.tensor(ignore_labels).to('cuda'))

        logits = logits[mask]
        labels = labels[mask]

        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)

        return loss

    
    def smooth_nll_loss(self, logits, labels, shift_labels=True, epsilon=0.1):
        
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        # padding_mask = labels.eq(self.ignore_index)

        ignore_labels = [-100, self.tokenizer.pad_token_id]
        padding_mask = torch.isin(labels, torch.tensor(ignore_labels).to('cuda'))


        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - epsilon) * nll_loss + epsilon * smoothed_loss
    
    def compute_loss(self, model, inputs, return_outputs=False):

        """
        Custom loss function: CrossEntropyLoss with optional return of model outputs.
        """
        
        labels = inputs.pop("labels")   

        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.smooth_nll_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, **kwargs):
        # Use a subset of the evaluation dataset if eval_dataset is provided or use self.eval_dataset
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # Randomly select a subset of indices for evaluation
        subset_indices = random.sample(range(len(eval_dataset)), min(self.eval_subset_size, len(eval_dataset)))
        subset_dataset = Subset(eval_dataset, subset_indices)

        # Call the original evaluate function with the subset dataset
        return super().evaluate(eval_dataset=subset_dataset, **kwargs)


training_args = TrainingArguments(
    output_dir="/home/ubuntu/volume/results",          # Directory to save checkpoints and logs
    overwrite_output_dir=True,       # Overwrite the content of the output directory
    num_train_epochs=3,              # Total number of training epochs
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    warmup_steps=500,                # Warmup steps for learning rate scheduler
    weight_decay=0.01,               # Weight decay for optimizer
    logging_dir="./logs",            # Directory to save logs
    logging_steps=100,                # Log every 10 steps
    eval_strategy="steps",     # Evaluate after every epoch
    eval_steps=1000,
    save_strategy="steps",           # Save model checkpoints after each epoch
    save_steps=1000,
    save_total_limit=1,              # Limit the total number of checkpoints
    load_best_model_at_end=True,     # Load the best model at the end of training
    metric_for_best_model="rouge-l",# Metric to determine the best model
    greater_is_better=True,          # Whether a higher metric is better
    learning_rate=5e-5,              # Initial learning rate
    report_to="tensorboard",         # Report to TensorBoard for visualization
    fp16=True,                       # Enable mixed precision training (faster on GPUs)
    remove_unused_columns=False,
    optim="sgd",
    # save_only_model=True,
)

# 8. Initialize Custom Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# checkpoint_path = "/home/ubuntu/volume/results/checkpoint-50000"

# 9. Train the Model
trainer.train()









