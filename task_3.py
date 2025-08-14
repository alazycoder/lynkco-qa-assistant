import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType


base_model = "Qwen/Qwen2-7B"
output_dir = "./models/lora-qwen-2-7b"
print(base_model)


tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# 加载数据集
def read_txt2dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return Dataset.from_dict({'text': texts})


def tokenize(examples):
    return tokenizer(examples["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=1024)


def add_labels(example):
    example['labels'] = example['input_ids']
    return example


custom_dataset = read_txt2dataset('./data/all_text.txt')
split_dataset = custom_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

tokenized_train_dataset = train_dataset.map(tokenize, batched=True).map(add_labels)
tokenized_eval_dataset = eval_dataset.map(tokenize, batched=True).map(add_labels)

model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32, trust_remote_code=True, device_map="auto")


# LoRA 配置
lora_config = LoraConfig(
    r=8,              # bottleneck的秩作低秩分解
    lora_alpha=16,    # LoRA scaling 参数
    target_modules=["q_proj","v_proj"],
    # target_modules=["W_pack"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

# 训练配置
training_args = TrainingArguments(
    output_dir=output_dir + "_output",
    auto_find_batch_size=True,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    save_strategy="steps",
    save_steps=100,
    eval_steps=50,
    logging_steps=10,
    num_train_epochs=2,
    save_total_limit=5,
    # report_to="tensorboard",
    # logging_dir='./logs/tensorboard'
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 关闭MLM，表示自回归语言建模
)

# 训练入口
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
)


trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
