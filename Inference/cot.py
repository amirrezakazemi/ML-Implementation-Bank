from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm

# model_path = "./saved_models/gpt-neo-1.3B"
# model_path = '/home/ubuntu/volume/results/checkpoint-50000/'
model_path = '/home/ubuntu/volume/meta-llama-3-8b-instruct/'
dataset_path = "/home/ubuntu/volume/GSM8K_test.jsonl"

gpu_device = "0"

output_path = os.path.join(os.path.dirname(dataset_path), "GSM8K_test_llama_instruction_prediction_8shot.jsonl")

model = AutoModelForCausalLM.from_pretrained(model_path, device_map=f"cuda:{gpu_device}")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

sample_q1 = "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
sample_q2 = "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
sample_q3 = " Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
sample_q4 = "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"
sample_q5 = "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"
sample_q6 = "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"
sample_q7 = "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"
sample_q8 = "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"

sample_a1 = "Step 1: There are 15 trees originally. Step 2: Then there were 21 trees after some more were planted. Step 3: So there must have been 21 - 15 = 6. Step 4: The answer is 6. #### 6"
sample_a2 = "Step 1: There are originally 3 cars. 2 more cars arrive. Step 2: 3 + 2 = 5. Step 4: The answer is 5. #### 5"
sample_a3 = "Step 1: Originally, Leah had 32 chocolates. Step 2: Her sister had 42. Step 3: So in total they had 32 + 42 = 74. Step 4: After eating 35, they had 74 - 35 = 39. Step 5: The answer is 39. #### 39"
sample_a4 = "Step 1: Jason started with 20 lollipops. Step 2: Then he had 12 after giving some to Denny. Step 3: So he gave Denny 20 - 12 = 8. Step 4: The answer is 8. #### 8"
sample_a5 = "Step 1: Shawn started with 5 toys. Step 2: If he got 2 toys each from his mom and dad, then that is 4 more toys. Step 3: 5 + 4 = 9. Step 4: The answer is 9. #### 9"
sample_a6 = "Step 1: There were originally 9 computers. Step 2: For each of 4 days, 5 more computers were added. Step 3: So 5 * 4 = 20 computers were added. Step 4: 9 + 20 is 29. Step 5: The answer is 29. #### 29"
sample_a7 = "Step 1: Michael started with 58 golf balls. Step 2: After losing 23 on Tuesday, he had 58 - 23 = 35. Step 3: After losing 2 more, he had 35 - 2 = 33 golf balls. Step 4: The answer is 33. #### 33"
sample_a8 = "Step 1: Olivia had 23 dollars. Step 2: 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. Step 3: So she has 23 - 15 dollars left. Step 4: 23 - 15 is 8. Step 5: The answer is 8. #### 8"

prompt = (
"Here are also examples of a mathematical question with the correct response."
f"##Question: {sample_q1}\n"
f"##Response: {sample_a1}\n\n"
f"##Question: {sample_q2}\n"
f"##Response: {sample_a2}\n\n"
f"##Question: {sample_q3}\n"
f"##Response: {sample_a3}\n\n"
f"##Question: {sample_q4}\n"
f"##Response: {sample_a4}\n\n"
f"##Question: {sample_q5}\n"
f"##Response: {sample_a5}\n\n"
f"##Question: {sample_q6}\n"
f"##Response: {sample_a6}\n\n"
f"##Question: {sample_q7}\n"
f"##Response: {sample_a7}\n\n"
f"##Question: {sample_q8}\n"
f"##Response: {sample_a8}\n\n"
"""For the following question, write a correct response with explaining your reasoning steps. 
You must start step i with 'Step i:' and the last step must be the final answer and end generation there. 
Let's think step by step.""" 
)

# Open the JSONL file and read it line by line
with open(dataset_path, 'r') as infile, open(output_path, 'w') as outfile:


    total_lines = sum(1 for _ in infile)
    infile.seek(0)  # Reset file pointer to the start of the file

    for line in tqdm(infile, total=total_lines, desc="Processing lines"):
        # Parse each line as JSON
        data = json.loads(line)

        question = data['query']

        sample = (
            f"{prompt}"
            f"###Question: {question}\n"
            "##Response:"
        )

        tokenized_sample = tokenizer(sample, return_tensors='pt')
        len_sample = len(tokenized_sample['input_ids'][0])

        output = model.generate(
            input_ids=tokenized_sample['input_ids'].to(f"cuda:{gpu_device}"),
            attention_mask=tokenized_sample['attention_mask'].to(f"cuda:{gpu_device}"),
            max_new_tokens=512,
            do_sample=True,  # Enable sampling
            temperature=1.0,  # Controls randomness
            top_p=0.9,        # Limits cumulative probability of token set
            eos_token_id=tokenizer.eos_token_id,  # Ensure the model ends at EOS
            pad_token_id=tokenizer.pad_token_id   # To avoid padding issues
        )

        decoded_output = tokenizer.decode(
            output[0, len_sample:], skip_special_tokens=True
        )

        output_data = {
            "question": question,
            "llm_solution": decoded_output,
            "gt_solution": data.get("response")
        }

        outfile.write(json.dumps(output_data) + '\n')

