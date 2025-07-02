import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

BASE_MODEL = "/pythia-2.8b"
FINETUNED_MODEL = "/pythia-2.8b-nli"
#  设置数据类型为 FP16
torch_dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token  # 显式设置pad_token

model_pretrained = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
model_finetuned = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
def compute_perplexity(model, tokenizer, text, device='cuda'): # gpt_ppl_train_ood_score
    model.eval()  # 设为评估模式
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        # outputs = model(input_ids, labels=input_ids)
        outputs = model(input_ids, labels=input_ids,use_cache=False)
        loss = outputs.loss  # CrossEntropyLoss 已经计算了对数似然

    perplexity = torch.exp(loss)  # 取指数得到 PPL
    return perplexity.item()

#  计算 OOD 评分
def compute_ood_score(text):
    log_p_pretrained = compute_perplexity(model_pretrained, tokenizer, text)
    log_p_finetuned = compute_perplexity(model_finetuned, tokenizer, text)
    # print(log_p_pretrained)
    # print(log_p_finetuned)
    # print(log_p_finetuned - log_p_pretrained)
    return log_p_finetuned - log_p_pretrained

input_file = '/mnli/train_select.jsonl'
output_file = "/mnli/train_pythia_ood.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

results = []

#  计算 OOD 评分并保存
# for example in tqdm(data, desc="Computing OOD Scores"):
#     text = example["Text"]
#     Label = example["Label"]
#     ood_score = compute_ood_score(text)
#
#     #  存储结果
#     result = {
#         "Text": text,
#         "Label":Label,
#         "OOD_Score": ood_score
#     }
#     results.append(result)
for example in tqdm(data, desc="Computing OOD Scores"):
    # Premise = example["Premise"]
    # Hypothesis = example["Hypothesis"]
    Premise = example["Premise"]
    Hypothesis = example["Hypothesis"]
    Label = example["Label"]
    text = f"{Premise} {Hypothesis}"

    ood_score = compute_ood_score(text)

    #  存储结果
    result = {
        "Premise": Premise,
        "Hypothesis": Hypothesis,
        "Label":Label,
        "OOD_Score": ood_score
    }
    results.append(result)
#  保存结果为 JSONL 文件
print(results)
with open(output_file, "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"OOD scores saved to: {output_file}")