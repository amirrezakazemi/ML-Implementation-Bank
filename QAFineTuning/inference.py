from transformers import AutoModelForCausalLM, AutoTokenizer


# model_path = "./saved_models/gpt-neo-1.3B"
model_path = './results/checkpoint-98000/'
dataset_path = "./saved_datasets/alpaca/"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()
# for name,param in model.named_parameters():
#     print(f'name {name}, {param.shape}')
tokenizer = AutoTokenizer.from_pretrained(model_path)

# train_dataset, eval_dataset = load_split_data(dataset_path)

# train_dataset = IFTDataset(train_dataset)

# sample = train_dataset[1]['instruction']

question = "James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?"

sample = (
    "Below is a mathematical question. Write a correct response with explaining your reasoning steps."
    f"###Question: {question}\n\n"
    "##Response:"
)

print(sample)

tokenized_sample = tokenizer(sample, return_tensors='pt')

output = model.generate(
    input_ids=tokenized_sample['input_ids'].to('cuda'),
    attention_mask=tokenized_sample['attention_mask'].to('cuda'),
    max_new_tokens=100,
    do_sample=True,  # Enable sampling
    temperature=0.7,  # Controls randomness
    top_p=0.9,        # Limits cumulative probability of token set
    eos_token_id=tokenizer.eos_token_id,  # Ensure the model ends at EOS
    pad_token_id=tokenizer.pad_token_id   # To avoid padding issues
)

decoded_output = tokenizer.batch_decode(
    output
)

print(decoded_output)

