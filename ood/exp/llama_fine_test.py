import torch as torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import LlamaForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
from tqdm import tqdm
import json
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from peft import LoraConfig, get_peft_model, PeftModel

model = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/transformer/pythia-2.8b-toxic", torch_dtype=torch.float16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/transformer/pythia-2.8b-toxic")

# 确保pad_token设置正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 加载测试数据
with open(
         'test.jsonl',
        'r', encoding='utf-8') as f:
    test_data = [json.loads(line) for line in f]

predictions = []
labels = []

for example in tqdm(test_data, desc="Generating Predictions"):
    # input_text = f"Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral. Text: {example['Text']} Prediction:"
    # input_text = f"Solve the NLI task. Options for entailment relationship: entailment, neutral, contradiction.\nPremise: \"{example['Premise']}\"  Hypothesis: \"{example['Hypothesis']}\"  Prediction:"
    input_text = f"Solve the toxic detection task. Options for toxicity: benign, toxic. \nText: {example['Text']} Prediction:"
    # input_text = f"Solve the NER task, identifying the Organization, Person, Location entities from given text.\nText: {example['Text']} // Entity:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )


    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(pred)
    sentiment = pred.split("Prediction:")[-1].strip().lower()
    print(sentiment)

    # 标签解析逻辑保持不变
    # if 'positive' in sentiment:
    #     pred_label = 1
    # elif 'negative' in sentiment:
    #     pred_label = 0
    # elif 'neutral' in sentiment:
    #     pred_label = 2
    # else:
    #     pred_label = -1
    if 'benign' in sentiment:
        pred_label = 0
        print(0)
    elif 'toxic' in sentiment:
        pred_label = 1
        print(1)
    else:
        pred_label = -1
    # if 'entailment' in sentiment:
    #     pred_label = 0
    # elif 'contradiction' in sentiment:
    #     pred_label = 1
    #     print(1)
    # elif 'neutral' in sentiment:
    #     pred_label = 2
    #     print(2)
    # else:
    #     pred_label = -1
    #     print(-1)
    predictions.append(pred_label)
    labels.append(example['Label'])


filtered_preds = [predictions[i] for i in predictions]
print(predictions)
print(len(predictions))
filtered_labels = [labels[i] for i in predictions]
# print(filtered_preds)
# print(len(filtered_preds))
print(labels)
correct = sum(p == r for p, r in zip(predictions, labels))
# 计算指标
print(correct)
# accuracy = accuracy_score(filtered_labels, filtered_preds)
accuracy = correct/len(predictions)
print(accuracy)
f1 = f1_score(labels, predictions, average='weighted')

print(f"\nValid samples: {len(filtered_preds)}/{len(predictions)}")
