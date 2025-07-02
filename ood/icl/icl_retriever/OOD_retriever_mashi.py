import json
import random

import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch.nn.functional as F
from tqdm import tqdm

def compute_mahalanobis_matrix_torch(X):
    cov_matrix = torch.cov(X.T)
    epsilon = 1e-3 * torch.eye(X.shape[1], device=X.device)  # 调整正则化强度
    cov_inv = torch.linalg.inv(cov_matrix + epsilon)

    X_diff = X[:, None, :] - X[None, :, :]
    M_dist_matrix = torch.einsum('ijk,kl,ijl->ij', X_diff, cov_inv, X_diff)
    return torch.sqrt(torch.clamp(M_dist_matrix, min=0))

class OODRetriever:
    def __init__(self, model_name="/Qwen3-1.7B-nli"):
        self.test_file = '/contract_nli/test_qwen3_1.7B_ood.jsonl'
        self.train_file = '/mnli/train_qwen3_1.7B_ood.jsonl'
        self.embedding_file = '/mnli/train_qwen3_1.7_embedding.jsonl'
        self.test_exp_file = '/contract_nli/test_qwen3_1.7B_ood.jsonl'
        self.train_by_label = {0: [], 1: [], 2: []}
        # self.train_by_label = {0: [], 1: []}
        self.train_ood_scores = []
        self.train_data = []
        self.combined_texts = []
        self.train_embeddings = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 初始化模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.bfloat16
            # torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

        # 加载数据并生成/读取嵌入
        self._load_train_data()

    def _batch_embedding(self, texts, batch_size=8):
        """批处理生成嵌入"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True)#,use_cache=False
            # last_hidden = outputs.hidden_states[-1]
            last_hidden = outputs.hidden_states[-2]
            batch_emb = last_hidden.mean(dim=1)
            embeddings.append(batch_emb)

            del inputs, outputs, last_hidden
            torch.cuda.empty_cache()

        return torch.cat(embeddings, dim=0)

    def _save_embeddings(self, embeddings):
        embeddings_list = embeddings.cpu().numpy().tolist()
        with open(self.embedding_file, 'w', encoding='utf-8') as f:
            for emb in embeddings_list:
                f.write(json.dumps(emb) + '\n')

    def _load_embeddings(self):
        with open(self.embedding_file, 'r', encoding='utf-8') as f:
            embeddings = [json.loads(line) for line in f]
        return torch.tensor(embeddings, device=self.device)

    def _load_train_data(self):
        """加载训练数据并预计算/读取嵌入"""
        with open(self.train_file, 'r', encoding='utf-8') as f:
            self.train_data = [json.loads(line) for line in f]
        print(f" 训练数据加载完成，共 {len(self.train_data)} 条数据")
        self.combined_texts = [
            # f"{sample['Text']}"
            f"{sample['Premise']}{sample['Hypothesis']}"
            for sample in self.train_data
        ]

        for idx, sample in enumerate(self.train_data):
            label = sample['Label']
            self.train_by_label[label].append(idx)
            self.train_ood_scores.append(sample['OOD_Score'])

        self.train_ood_scores = torch.tensor(self.train_ood_scores, device=self.device)

        if os.path.exists(self.embedding_file):
            print(f" 正在从 {self.embedding_file} 加载训练集嵌入...")
            self.train_embeddings = self._load_embeddings()
        else:
            print(" 未找到保存的嵌入，开始生成...")
            self.train_embeddings = self._batch_embedding(self.combined_texts)
            self._save_embeddings(self.train_embeddings)

    def get_rtr_idx_list(self,ood_number):
        with open(self.test_file, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]

        # test_combined_texts = [sample['Text'] for sample in test_data]
        test_combined_texts = [f"{sample['Premise'][:128]}{sample['Hypothesis'][:128]}" for sample in test_data]
        # test_combined_texts = [f"{sample['Premise']}{sample['Hypothesis']}" for sample in test_data]
        test_embeddings = self._batch_embedding(test_combined_texts)

        rtr_idx_list = []
        for test_idx, test_sample in enumerate(tqdm(test_data, desc="Processing Test Samples", leave=False)):
            current_emb = test_embeddings[test_idx].unsqueeze(0).to(self.device)
            selected_indices = {0: [], 1: [], 2: []}
            remaining_top_global_indices = {0: [], 1: [], 2: []}
            remaining_top_sim_scores = {0: np.array([]), 1: np.array([]), 2: np.array([])}
            # selected_indices = {0: [], 1: []}
            # remaining_top_global_indices = {0: [], 1: []}
            # remaining_top_sim_scores = {0: np.array([]), 1: np.array([])}
            selected_embeddings = []

            for label in range(3):
            # for label in range(2):
                label_indices = self.train_by_label[label]
                candidate_ood_scores = self.train_ood_scores[label_indices]

                # 按照 OOD 分数（越小越好）排序
                top1000_ood_idx = torch.topk(-candidate_ood_scores, k=ood_number).indices
                top1000_global_indices = [label_indices[i.item()] for i in top1000_ood_idx]
                selected_ood_embeddings = self.train_embeddings[top1000_global_indices]


                similarities = F.cosine_similarity(current_emb, selected_ood_embeddings).detach().cpu().numpy()
                top5_sim_idx = np.argsort(similarities)[-ood_number:]
                top5_global_indices = [top1000_global_indices[i] for i in top5_sim_idx]
                top5_sim_scores = similarities[top5_sim_idx]

                # 第一轮选一个最相似的
                best_idx = np.argmax(top5_sim_scores)
                best_global_idx = top5_global_indices.pop(best_idx)
                selected_indices[label].append(best_global_idx)

                # 剩余留作第二轮使用
                remaining_top_global_indices[label] = top5_global_indices
                remaining_top_sim_scores[label] = np.delete(top5_sim_scores, best_idx)
            #
            for label in range(3):
            # for label in range(2):
                selected_embeddings.append(torch.stack([self.train_embeddings[idx] for idx in selected_indices[label]]))
            selected_embeddings = torch.cat(selected_embeddings)
            M_matrix = compute_mahalanobis_matrix_torch(selected_embeddings)
            mean_mahalanobis = torch.mean(
                M_matrix[torch.triu_indices(M_matrix.shape[0], M_matrix.shape[1], offset=1)]
            )

            # 第二轮从剩下的 top4 选
            # for label in range(3):

            for label in range(2):
                while len(selected_indices[label]) < 3 and remaining_top_global_indices[label]:
                    best_idx = np.argmax(remaining_top_sim_scores[label])
                    best_global_idx = remaining_top_global_indices[label].pop(best_idx)
                    remaining_top_sim_scores[label] = np.delete(remaining_top_sim_scores[label], best_idx)

                    new_embedding = self.train_embeddings[best_global_idx].unsqueeze(0)
                    new_embeddings = torch.cat([selected_embeddings, new_embedding], dim=0)
                    new_M_matrix = compute_mahalanobis_matrix_torch(new_embeddings)
                    new_mean_mahalanobis = torch.mean(
                        new_M_matrix[torch.triu_indices(new_M_matrix.shape[0], new_M_matrix.shape[1], offset=1)]
                    )

                    if new_mean_mahalanobis >= mean_mahalanobis:
                    # if new_mean_mahalanobis < mean_mahalanobis:
                        selected_indices[label].append(best_global_idx)
                        selected_embeddings = new_embeddings
                        mean_mahalanobis = new_mean_mahalanobis

            final_selected = sum(selected_indices.values(), [])
            random.shuffle(final_selected)
            rtr_idx_list.append(final_selected)

        return rtr_idx_list

    def format_predictions(self, idx_list):
        label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
        return ''.join(
            f"Text: {self.train_data[idx]['Text']} Prediction: {label_map[self.train_data[idx]['Label']]}\n"
            for idx in idx_list
        )




if __name__ == "__main__":
    retriever = OODRetriever()
    ood_numbers = [300,1000]
    for ood_number in ood_numbers:
        rtr_idx_list = retriever.get_rtr_idx_list(ood_number=ood_number)
        output_file = f'/contract_nli/qwen_mashi_sim3_ood{ood_number}.jsonl'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(rtr_idx_list, f, ensure_ascii=False, indent=4)
        print(f" OOD数量 {ood_number} 的结果已保存到 {output_file}")

