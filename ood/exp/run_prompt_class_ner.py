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

from openicl.icl_retriever.icl_ner_retriever import NerBERTRetriever

file_path = '/ener/gemma_mashi_sim3_ood800.jsonl'
print(file_path)
with open(file_path, 'r', encoding='utf-8') as f:
    g_ice_idx_list = json.load(f)


model_name = "gemma-2-2b"
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

def format_predictions(idx_list,train_dataset):
    formatted_predictions = []
    for idx in idx_list:
        text = train_dataset[idx]["Text"]
        entity = train_dataset[idx]["Entity"]
        person = entity["person"]
        person = ','.join(person)
        location = entity["location"]
        location = ','.join(location)
        organization = entity["organization"]
        organization = ','.join(organization)
        formatted_predictions.append(f"Text: {text} Entity: Organization: {organization} || Person: {person} || Location: {location}\n")
    return ''.join(formatted_predictions)

from tqdm import tqdm  # 引入 tqdm

def main(train_path, test_path, model_path, sentence_model_path, input_columns_name, output_columns_name,
         ice_num, candidate_num, select_time, batch_size, seed, output_json_filepath):
    # load dataset
    combined_dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
    train_dataset = combined_dataset["train"]
    test_dataset = combined_dataset["test"]
    accelerator = Accelerator()
    data = DatasetReader(combined_dataset, input_columns=input_columns_name, output_column=output_columns_name)
    bert_retriever = NerBERTRetriever(data, ice_num=ice_num, index_split='train', test_split='test',
                                      accelerator=accelerator)
    ice_idx_list = bert_retriever.retrieve()
    ice = []

    predictions = []
    count = 0.0

    for idx in tqdm(range(len(test_dataset)), desc="Processing test samples", unit="sample"):
        # ice_item = bert_retriever.generate_ice(ice_idx_list[idx], ice_template=template)
        ice_item = format_predictions(ice_idx_list[idx],train_dataset)
        torch.cuda.empty_cache()
        text = test_dataset[idx]["Text"]
        entity = test_dataset[idx]["Entity"]
        entities = []
        for item in entity["organization"]:
            entities.append("O-"+item)
        for item in entity["person"]:
            entities.append("P-"+item)
        for item in entity["location"]:
            entities.append("L-"+item)
        print(entities)
        prompt = f"Solve the NER task, identifying the Organization, Person, Location entities from given text.\n{ice_item}Text: {text} Entity:"
        print(prompt)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True , padding=True,max_length=980)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate batch results
        outputs = model.generate(
            **inputs,
            max_new_tokens=23,  # 控制生成的最大token数量，而不是总长度
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )

        # Extract the explanation (predicted label)
        batch_explanations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        explanation = batch_explanations[0]
        # print(explanation)
        lines = [line.strip() for line in explanation.split('\n') if line.strip()]
        entity_lines = [line for line in lines if "Entity:" in line]
        pre = ""
        if len(entity_lines) >= 4:
            tenth_entity = entity_lines[3]  # Python索引从0开始，第10个是索引9
            entity_content = tenth_entity.split("Entity:", 1)[1].strip()

            pre = entity_content
        # pre = explanation.replace(prompt,"")
        result = pre.split("Text", 1)[0].lstrip()
        print(result)
        parts = [part.strip() for part in result.split('||')]
        result_last = []
        for part in parts:
            # 分割键值对（只按第一个冒号分割）
            split_part = part.split(':', 1)
            if len(split_part) != 2:
                continue  # 跳过格式错误的部分

            key, value = split_part
            key = key.strip()
            values = [v.strip() for v in value.split(',')]  # 按逗号分隔多个值

            # 确定前缀
            if key == 'Organization':
                prefix = 'O'
            elif key == 'Person':
                prefix = 'P'
            elif key == 'Location':
                prefix = 'L'
            else:
                continue  # 跳过未知类型

            # 为每个值添加前缀
            for v in values:
                if v:  # 忽略空值
                    result_last.append(f"{prefix}-{v}")

        # 组合结果
        print("result",result_last)
        set1 = set(entities)  # {'O-None', 'P-None', 'L-None'}
        set2 = set(result_last)  # {'O-Paramount', 'P-None', 'L-None'}

        # 计算重复元素数量和重复率
        common_elements = set1 & set2  # 交集
        repetition_rate = len(common_elements) / max(len(set1), len(set2))  # 避免除以零

        print(f"去重后的重复率: {repetition_rate:.3%}")
        count = count + repetition_rate


    accuracy = count/len(test_dataset)
    print(f"Accuracy: {accuracy}")

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
                     "nli": ['Premise', 'Hypothesis'],
                     "ner":['Text']
                     }

    output_columns = {
                      "amazon": 'Label',
                      'td': 'Label',
                      "nli": 'Label',
                      "ner":'Entity'
                      }

    test_split = {
        "amazon": 'test',
        'td': 'test',
        "nli": 'validation',
        "ner":'test'
    }
    task_names = ["ner"]
    model_names = ['gpt2']
    seeds = [1, 43, 666]
    # set the model and dataset path
    # model_dir = 'C:\\Users\\bort\\.cache\\huggingface\\transformers\\'
    model_dir = '/root/autodl-tmp/huggingface/transformers/'
    sentence_transformer_path = '/root/autodl-tmp/huggingface/transformers/all-mpnet-base-v2'
    data_dir = ''

    for model_name in model_names:
        model_path = model_dir + model_name
        sentence_model_path = sentence_transformer_path

        for seed in seeds:
            for task_name in task_names:
                train_path = '/fewnerd/train.jsonl' # train_filtered
                test_name = test_split[task_name]
                test_path = '/wnut/test.jsonl' # test_filtered
                ice_num = 3
                output_json_filepath = '/results/' + model_name + '/' + task_name
                import os

                os.makedirs(output_json_filepath, exist_ok=True)

                batch_size = 10

                candidate_num = 30
                select_time = 10

                main(train_path, test_path, model_path, sentence_model_path,
                     input_columns[task_name], output_columns[task_name], ice_num, candidate_num, select_time,
                     batch_size, seed, output_json_filepath)
