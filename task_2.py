import os
import json
import time
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

from score import ScoreModel


model_name = "baichuan-inc/Baichuan2-7B-Chat"
torch_type = torch.float16
batch_size = 2
gpus = "0,1"
max_new_tokens = 60

print(f"start time: {datetime.now()}")
print(f"model_name: {model_name}, torch_type: {torch_type}, batch_size: {batch_size}, gpus: {gpus}")
print(f"max_new_tokens: {max_new_tokens}")

os.environ["CUDA_VISIBLE_DEVICES"] = gpus
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_type, trust_remote_code=True, device_map="auto")
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id
model.eval()
score_model = ScoreModel()
# time.sleep(30) # 让 prometheus 采集 GPU metrics

bench_data = json.load(open("data/bench_data.json"))
gold_data = json.load(open("data/gold.json"))
print(f"number of tests: {len(bench_data)}, {len(gold_data)}")
all_ground_truth = [d["answer"] for d in gold_data]
all_keywords = [d["keywords"] for d in gold_data]
all_prompts = bench_data.values()
all_tests = [(prompt, answer, keyword) for prompt, answer, keyword in zip(all_prompts, all_ground_truth, all_keywords)]



def chunked(iterable, n):
    """每 n 个分一组"""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


total_time = 0
total_score = 0.0

for chunk in chunked(all_tests, batch_size):
    prompts = [c[0] for c in chunk]
    ground_truths = [c[1] for c in chunk]
    keywords = [c[2] for c in chunk]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)  # 批输入编码到cuda

    input_ids = inputs['input_ids']
    print(f"shape of input_ids: {input_ids.shape}")

    st = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    total_time += (time.time() - st) * 1000

    print(f"shape of generated_ids: {generated_ids.shape}")
    answers = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

    for prompt, ground_truth, keyword, answer in zip(prompts, ground_truths, keywords, answers): 
        print(f"prompt: {prompt}")
        print(f"ground truth: {ground_truth}")
        print(f"keywords: {keyword}")
        print(f"answer: {answer}")
        score, semantic_score, keyword_score = score_model.get_score(ground_truth, keyword, answer)
        total_score += score
        print(f"score: {score}, semantic_score: {semantic_score}, keyword_score: {keyword_score}")
        print()

print(f"model generate all batches time: {total_time:.1f} ms")
print(f"average score: {total_score/len(all_tests)}")
print(f"end time: {datetime.now()}")
