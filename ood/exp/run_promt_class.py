import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from openicl import PromptTemplate
from openicl import DatasetReader
from openicl import RandomRetriever, BM25Retriever, ConERetriever, TopkRetriever, PPLInferencer, AccEvaluator
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from torch import bfloat16

from openicl.icl_retriever.icl_colBert_retriever import ColBERTRetriever
import json

file_path = '/wanli/qwen_mashi_sim3_ood300.jsonl'

with open(file_path, 'r', encoding='utf-8') as f:
    g_ice_idx_list = json.load(f)

model_name = "/root/autodl-fs/transformer/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True).half()

# 设置填充标记为结束标记
tokenizer.pad_token = tokenizer.eos_token  # 设置填充标记为结束标记（eos_token）
model.config.pad_token_id = tokenizer.eos_token_id  # 设置pad_token_id为eos_token_id

# 检查GPU是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # 将模型移到GPU或CPU

def calculate_accuracy(predictions, references):
    correct = sum(p == r for p, r in zip(predictions, references))
    total = len(references)
    return correct / total if total > 0 else 0.0

def remove_label(sample):
    sample["Label"] = ""  # 将 Label 字段替换为空格
    return sample

def format_predictions(idx_list,train_dataset):
    #sentiment analyis
    # label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    # formatted_predictions = []
    #
    # for idx in idx_list:
    #         text = train_dataset[idx]["Text"]
    #         label = train_dataset[idx]["Label"]
    #         label_str = label_map[label]
    #         formatted_predictions.append(f"Text: {text} Prediction: {label_str}\n")

    #toxic detection
    # label_map = {0: 'benign', 1: 'toxic'}
    # formatted_predictions = []
    # # print(idx_list)
    #
    # for idx in idx_list:
    #     text = train_dataset[idx]["Text"]
    #     label = train_dataset[idx]["Label"]
    #     label_str = label_map[label]
    #     formatted_predictions.append(f"Text: {text} Prediction: {label_str}\n")

    label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
    formatted_predictions = []

    for idx in idx_list:
        Premise = train_dataset[idx]["Premise"][:128]
        Hypothesis = train_dataset[idx]["Hypothesis"][:128]
        label = train_dataset[idx]["Label"]
        label_str = label_map[label]
        formatted_predictions.append(f"Premise: {Premise} Hypothesis: {Hypothesis} Prediction: {label_str}\n")

    return ''.join(formatted_predictions)

from tqdm import tqdm  # 引入 tqdm

def main(template, train_path, test_path, model_path, sentence_model_path, input_columns_name, output_columns_name,
         ice_num, candidate_num, select_time, batch_size, seed, output_json_filepath):
    # load dataset
    combined_dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
    train_dataset = combined_dataset["train"]
    test_dataset = combined_dataset["test"]
    labels = [item["Label"] for item in test_dataset]
    data = DatasetReader(combined_dataset, input_columns=input_columns_name, output_column=output_columns_name)

    # Different retrieval strategies
    print("Start inference....")
    ice_idx_list = g_ice_idx_list

    ice = []

    predictions = []

    for idx in tqdm(range(len(test_dataset)), desc="Processing test samples", unit="sample"):
        ice_item = format_predictions(ice_idx_list[idx],train_dataset)
        torch.cuda.empty_cache()
        # text = test_dataset[idx]["Text"]
        # # prompt = f"Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral.\n{ice_item}Text: {text} Prediction:"
        # prompt = f"Solve the toxic detection task. Options for toxicity: benign, toxic.\n{ice_item}Text: {text} Prediction:"
        text1 = test_dataset[idx]["Premise"]
        text2 = test_dataset[idx]["Hypothesis"]
        prompt = f"Solve the NLI task. Options for entailment relationship: entailment, neutral, contradiction.\n{ice_item}Premise: {text1}  Hypothesis: {text2} Prediction:"

        # print(prompt)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True , padding=True,max_length=1020)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate batch results
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,  # 控制生成的最大token数量，而不是总长度
            pad_token_id=tokenizer.eos_token_id,
            # use_cache=False
        )

        # Extract the explanation (predicted label)
        batch_explanations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        explanation = batch_explanations[0]
        sentiment = explanation.split("Prediction:")[-1].strip().lower()
        # print(sentiment)
        # if 'positive' in sentiment:
        #     predictions.append(1)
        #     # print(1)
        # elif 'negative' in sentiment:
        #     predictions.append(0)
        #     # print(0)
        # elif 'neutral' in sentiment:
        #     predictions.append(2)
        #     # print(2)
        # else:
        #     predictions.append(-1)
        #
        # if 'benign' in sentiment:
        #     predictions.append(0)
        #     # print(0)
        # elif 'toxic' in sentiment:
        #     predictions.append(1)
        #     # print(1)
        # else:
        #     predictions.append(-1)

        if 'entailment' in sentiment:
            predictions.append(0)
            # print(0)
        elif 'contradiction' in sentiment:
            predictions.append(1)
            # print(1)
        elif 'neutral' in sentiment:
            predictions.append(2)
            # print(2)
        else:
            predictions.append(-1)
            # print(-1)

    # Save predictions to a JSON file
    accuracy = calculate_accuracy(predictions, labels)
    print(f"Accuracy: {accuracy}")
    with open(f"{output_json_filepath}/predictions.json", "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    # print(f"Predictions saved to {output_json_filepath}/predictions.json")



if __name__ == '__main__':
    import nltk

    print(nltk.data.path)
    amazon_tp_dict = {
        0: "</E>Text: </text> Prediction: negative",
        1: "</E>Text: </text> Prediction: positive",
        2: "</E>Text: </text> Prediction: neutral"
    }
    amazon_template = PromptTemplate(amazon_tp_dict, {'Text': '</text>'}, ice_token='</E>')

    td_tp_dict = {
        0: "</E>Text: </text> Prediction: benign",
        1: "</E>Text: </text> Prediction: toxic"
    }
    td_template = PromptTemplate(td_tp_dict, {'Text': '</text>'}, ice_token='</E>')

    nli_tp_dict = {
        0: "</E>Premise: </text1> Hypothesis:</text> Prediction: entailment",
        1: "</E>Premise: </text1> Hypothesis:</text> Prediction: contradiction",
        2: "</E>Premise: </text1> Hypothesis:</text> Prediction: neutral"
    }
    nli_template = PromptTemplate(nli_tp_dict, {'Premise': '</text1>', 'Hypothesis': '</text>'}, ice_token='</E>')
    templates = {
                 "amazon": amazon_template,
                 'td': td_template,
                 "nli": nli_template
                 }

    input_columns = {
                     "amazon": ["Text"],
                     'td': ['Text'],
                     "nli": ['Premise', 'Hypothesis']
                     }

    output_columns = {
                      "amazon": 'Label',
                      'td': 'Label',
                      "nli": 'Label'
                      }

    test_split = {
        "amazon": 'test',
        'td': 'test',
        "nli": 'validation'
    }
    task_names = ['nli']
    model_names = ['gpt2']
    seeds = [1, 43, 666]

    # set the model and dataset path
    model_dir = '/transformers/'
    sentence_transformer_path = ''
    data_dir = ''

    for model_name in model_names:
        model_path = model_dir + model_name
        sentence_model_path = sentence_transformer_path

        for seed in seeds:
            for task_name in task_names:
                # train_path = '/civil_comments/train_select.jsonl' # train_filtered
                # train_path = '/amazon/train_select.jsonl'  # train_filtered
                train_path = '/mnli/train_select.jsonl'  # train_filtered
                test_name = test_split[task_name]
                # test_path = '/toxigen/test.jsonl' # test_filtered
                # test_path = '/dynasent/test_filtered.jsonl'  # test_filtered
                test_path = '/wanli/test_select.jsonl'
                ice_num = 9
                # output_json_filepath = '/root/autodl-tmp/llm_code/revisit_demon_selection_in_ICL-main/exp/ToxicDetection/civil_comments/result_sim2' + '/' + task_name
                output_json_filepath = '/results/' + model_name + '/' + task_name
                import os

                os.makedirs(output_json_filepath, exist_ok=True)

                batch_size = 10

                candidate_num = 30
                select_time = 10

                main(templates[task_name], train_path, test_path, model_path, sentence_model_path,
                     input_columns[task_name], output_columns[task_name], ice_num, candidate_num, select_time,
                     batch_size, seed, output_json_filepath)
