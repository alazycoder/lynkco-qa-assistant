import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model_name = "Qwen/Qwen2-7B"
# model_name = "baichuan-inc/Baichuan2-7B-Chat"
model_name = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float32, token="xxx")
