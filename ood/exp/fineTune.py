import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, AutoTokenizer, \
    AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import json

# 模型路径
# MODEL_NAME = "/root/autodl-tmp/meta-llama/Llama-3.2-3B"
# MODEL_NAME = "/root/autodl-fs/transformer/Qwen3-1.7B"
# MODEL_NAME = "/root/autodl-fs/transformer/pythia-2.8b"
MODEL_NAME = "/root/autodl-fs/google/gemma-2-2b"
# MODEL_NAME = "/root/autodl-fs/Qwen/Qwen2.5-3B"
# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

# lora_config = LoraConfig(
#     r=16,  # 降低Rank值
#     lora_alpha=32,  # alpha = 2*r
#     lora_dropout=0.1,
#     target_modules=["q_proj", "v_proj"],  # 仅适配关键层
#     task_type="CAUSAL_LM",
# )
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "dense"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

train_data = load_data('train.jsonl')
test_data = load_data('test.jsonl')

#定义数据集格式
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        example = self.data[idx]
        text = example['Text']
        entities = example['Entity']

        # 格式化实体预测（与format_predictions保持一致）
        def format_entity(entity_list):
            joined = ', '.join(entity_list) if entity_list else "None"
            return joined if joined.strip() else "None"

        person = format_entity(entities.get("person", []))
        location = format_entity(entities.get("location", []))
        organization = format_entity(entities.get("organization", []))

        # 构建NER任务prompt（参考format_predictions风格）
        prompt = (
            f"Solve the NER task, identifying the Organization, Person, Location entities from given text.\n"
            f"Text: {text}  Entity:"
        )

        # 完整答案（用于teacher forcing）
        answer = (
            f"Organization: {organization} || Person: {person} || Location: {location}\n"
        )

        # Tokenization
        # full_text = prompt + " " + answer
        #
        # inputs = tokenizer(full_text, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        #
        # return {
        #     "input_ids": inputs["input_ids"].squeeze(0),
        #     "attention_mask": inputs["attention_mask"].squeeze(0),
        #     "labels": inputs["input_ids"].squeeze(0),
        # }
        full_text = prompt + answer
        inputs = tokenizer(full_text, padding="max_length", max_length=256, truncation=True, return_tensors="pt")

        # mask prompt 部分的 label，使其不会被训练
        labels = inputs["input_ids"].clone()
        prompt_len = len(tokenizer(prompt)["input_ids"])
        labels[:, :prompt_len] = -100  # mask掉 prompt 部分

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


    # def __getitem__(self, idx):
    #     example = self.data[idx]
    #     # input_text = f"Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral. Text: \"{example['Text']}\" Prediction:"
    #     # # input_text = f"""判断以下文本的情感倾向（positive/negative/neutral）：文本内容：{example['Text']}答案："""  # 显式要求文本输出
    #     # label_map = {1: "positive", 0: "negative", 2: "neutral"}
    #     input_text = f"Solve the toxic detection task. Options for toxicity: benign, toxic. Text: \"{example['Text']}\" Prediction:"
    #     label_map = {0: "benign", 1: "toxic"}
    #     label_text = label_map.get(example["Label"])
    #
    #     full_text = input_text + " " + label_text
    #
    #     # Tokenization
    #     inputs = tokenizer(full_text, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    #
    #
    #     return {
    #         "input_ids": inputs["input_ids"].squeeze(0),
    #         "attention_mask": inputs["attention_mask"].squeeze(0),
    #         "labels": inputs["input_ids"].squeeze(0),  # 让 labels 和 input_ids 对齐
    #     }

        # example = self.data[idx]
        # input_text = f"Solve the NLI task. Options for entailment relationship: entailment, neutral, contradiction.\nPremise: \"{example['Premise']}\"  Hypothesis: \"{example['Hypothesis']}\"  Prediction:"
        # label_map = {1: "contradiction", 0: "entailment", 2: "neutral"}
        # # input_text = f"Solve the toxic detection task. Options for toxicity: benign, toxic. Text: \"{example['Text']}\" Prediction:"
        # # label_map = {0: "benign", 1: "toxic"}
        # label_text = label_map.get(example["Label"])
        #
        # full_text = input_text + " " + label_text
        #
        # # Tokenization
        # inputs = tokenizer(full_text, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        # return {
        #     "input_ids": inputs["input_ids"].squeeze(0),
        #     "attention_mask": inputs["attention_mask"].squeeze(0),
        #     "labels": inputs["input_ids"].squeeze(0),  # 让 labels 和 input_ids 对齐
        # }
        # #

train_dataset = SentimentDataset(train_data)
test_dataset = SentimentDataset(test_data)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="/gemma-2-2b-ner/results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='/google/gemma-2-2b-ner/logs',
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="tensorboard",
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 开始微调
trainer.train()

# 保存微调后的模型
model = model.merge_and_unload()
model.save_pretrained("/gemma-2-2b-ner")
tokenizer.save_pretrained("/gemma-2-2b-ner")


