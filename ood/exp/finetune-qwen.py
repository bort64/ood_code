import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import json

# 模型路径
MODEL_NAME = "Qwen3-4B"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True).cuda()

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    r=16,  # 恢复默认秩
    lora_alpha=16,
    lora_dropout=0.05,  # 降低dropout
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 加载数据集
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data
#
train_data = load_data('train.jsonl')
test_data = load_data('test.jsonl')

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, data):  # 关键点：双下划线且正确缩进
        super().__init__()  # 显式调用父类初始化（可选但推荐）
        self.data = data

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        example = self.data[idx]
        input_text = f"""判断以下文本的情感倾向（positive/negative/neutral）：文本内容：{example['Text']}答案："""  # 显式要求文本输出

        #使用文本标签代替数值
        label_text = ["negative", "neutral", "positive"][example["Label"]]
        # input_text = f"Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral.\nText: {example['Text']} Prediction:"
        # label_text = ["benign", "toxic"][example["Label"]]

        # 完整序列编码
        full_text = input_text + label_text
        inputs = tokenizer(full_text, padding="max_length", max_length=256,
                           truncation=True, return_tensors="pt")

        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        prefix_len = len(tokenizer(input_text)["input_ids"]) - 1  # 计算前缀长度
        labels[:prefix_len] = -100  # mask非答案部分

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }

train_dataset = SentimentDataset(train_data)
test_dataset = SentimentDataset(test_data)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="/Qwen3-4B-sentiment/results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir='/Qwen3-4B-sentiment/logs',
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="tensorboard",
    learning_rate=2e-5,
    label_smoothing_factor=0.1,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True,
    load_best_model_at_end=True
)

# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 开始微调
trainer.train()

# 保存微调后的模型
model.save_pretrained("/Qwen3-4B-sentiment")
tokenizer.save_pretrained("/Qwen3-4B-sentiment")
