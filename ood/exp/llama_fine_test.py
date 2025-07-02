# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# model_name = "/root/autodl-tmp/meta-llama/llama3-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
#
# model.eval()
#
#
# input_text = "This Hewlett Packard cartridge works fine, as has all previous same cartridges. I've always picked genuine catridges for all my printers"
# inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
#
# output = model.generate(**inputs, max_new_tokens=50)
# print(tokenizer.decode(output[0], skip_special_tokens=True))
#
import torch as torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import LlamaForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
from tqdm import tqdm
import json
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from peft import LoraConfig, get_peft_model, PeftModel

# 加载模型和tokenizer
# model = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/google/gemma-2-2b-sentiment").cuda()
# tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/google/gemma-2-2b-sentiment")
model = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/transformer/pythia-2.8b-toxic", torch_dtype=torch.float16, trust_remote_code=True).cuda()
# model = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/Qwen/Qwen2.5-7B-Instruct").cuda()
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/transformer/pythia-2.8b-toxic")

# model = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/transformer/gpt2-xl-ner").cuda()
# tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/transformer/gpt2-xl-ner")
# model = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/transformer/Qwen3-4B").cuda()
# tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/transformer/Qwen3-4B")

# MODEL_NAME = "/root/autodl-fs/transformer/gpt2-xl"
#
# # ✅ 加载tokenizer和模型
# tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
# tokenizer.pad_token = tokenizer.eos_token
#
# model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).cuda()
# #
# base_model = LlamaForCausalLM.from_pretrained("/root/autodl-tmp/meta-llama/llama3-sentiment").cuda()
# model = PeftModel.from_pretrained(base_model, "/root/autodl-tmp/meta-llama/llama3-sentiment/lora_adapter").cuda()

# 确保pad_token设置正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 加载测试数据
with open(
        # '/root/autodl-tmp/llm_code/revisit_demon_selection_in_ICL-main/exp/NameEntityRecognition/wnut/test.jsonl',
        # '/root/autodl-tmp/llm_code/revisit_demon_selection_in_ICL-main/exp/NaturalLanguageInference/anli/test_select.jsonl',
         '/root/autodl-tmp/llm_code/revisit_demon_selection_in_ICL-main/exp/ToxicDetection/implicit_hate/test.jsonl',
        # '/root/autodl-tmp/llm_code/revisit_demon_selection_in_ICL-main/exp/classification/SentimentAnalysis/sst5/test_filtered.jsonl',
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

    # 修改后的generate调用
    # outputs = model.generate(
    #      ** inputs,
    #      max_new_tokens = 3,
    #      pad_token_id = tokenizer.pad_token_id,
    #      eos_token_id = tokenizer.eos_token_id,
    #      do_sample=False,
    #      # use_cache=False
    # )
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
# print(f"Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")

# import torch
# from transformers import LlamaForCausalLM, AutoTokenizer
#
# model = LlamaForCausalLM.from_pretrained("/root/autodl-tmp/meta-llama/llama3-sentiment")
# for name, _ in model.named_parameters():
#     print(name)
# print(model)