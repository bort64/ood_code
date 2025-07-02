#  模型路径配置
BASE_MODEL = "/gemma-2-2b"
FINETUNED_MODEL = "/gemma-2-2b-ner"  # 替换为实际微调模型路径

# 设置数据类型为 FP16
torch_dtype = torch.float16

# 加载GPT-2专用tokenizer（修改点2）
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token  # 显式设置pad_token

# 加载预训练和微调模型（修改点3）
model_pretrained = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
model_finetuned = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()

def compute_perplexity(model, tokenizer, text, device='cuda'): # gpt_ppl_train_ood_score
    model.eval()  # 设为评估模式
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        # outputs = model(input_ids, labels=input_ids)
        outputs = model(input_ids, labels=input_ids,use_cache=False)
        loss = outputs.loss

    perplexity = torch.exp(loss)  # 取指数得到 PPL
    return perplexity.item()

#  计算 OOD 评分（保持原逻辑）
def compute_ood_score(text):
    log_p_pretrained = compute_perplexity(model_pretrained, tokenizer, text)
    log_p_finetuned = compute_perplexity(model_finetuned, tokenizer, text)
    return log_p_finetuned - log_p_pretrained

input_file = "train.jsonl"
output_file = "gemma_train_ood.jsonl"


with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

results = []

#  计算 OOD 评分并保存
for example in tqdm(data, desc="Computing OOD Scores"):
    text = example["Text"]
    entity = example["Entity"]
    ood_score = compute_ood_score(text)

    #  存储结果
    result = {
        "Text": text,
        "Entity":entity,
        "OOD_Score": ood_score
    }
    results.append(result)

print(results)
with open(output_file, "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"OOD scores saved to: {output_file}")