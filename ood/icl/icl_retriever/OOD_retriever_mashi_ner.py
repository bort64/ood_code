import json
import random

import numpy as np
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
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
    def __init__(self, model_name="gemma-2-2b-ner",
                 exp_file_path="/fewnerd/gemma_train_ood.jsonl"):
            self.test_file = '/ener/gemma_test_ood.jsonl'
            self.train_file = '/fewnerd/gemma_train_ood.jsonl'
            self.exp_file = exp_file_path
            self.embedding_file = '/fewnerd/gemma_train_embedding.jsonl'
            self.test_exp_file = '/ener/gemma_test_ood.jsonl'
            self.train_ood_scores = []
            self.train_data = []
            self.combined_texts = []
            self.train_embeddings = None

            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Initialize RoBERTa model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

            # Load data and generate embeddings
            self._load_train_data()

    def _batch_embedding(self, texts, batch_size=8):
            """Generate embeddings in batches using RoBERTa"""
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True,
                    add_special_tokens=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Use mean pooling of the last hidden states
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']

                # Expand attention mask for broadcasting
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_emb = sum_embeddings / sum_mask

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
            """Load training data and precompute/read embeddings"""
            with open(self.train_file, 'r', encoding='utf-8') as f:
                self.train_data = [json.loads(line) for line in f]

            print(f" Training data loaded, total {len(self.train_data)} samples")

            self.combined_texts = [
                f"{sample['Text']}"
                for sample in self.train_data
            ]

            for idx, sample in enumerate(self.train_data):
                self.train_ood_scores.append(sample['OOD_Score'])

            self.train_ood_scores = torch.tensor(self.train_ood_scores, device=self.device)

            if os.path.exists(self.embedding_file):
                print(f" Loading embeddings from {self.embedding_file}...")
                self.train_embeddings = self._load_embeddings()
            else:
                print(" No saved embeddings found, generating new ones...")
                self.train_embeddings = self._batch_embedding(self.combined_texts)
                self._save_embeddings(self.train_embeddings)
    def get_rtr_idx_list(self, ood_number):
        with open(self.test_file, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]

        # 批量获取测试集嵌入
        test_texts = [sample['Text'] for sample in test_data]
        test_embeddings = self._batch_embedding(test_texts)

        rtr_idx_list = []
        for test_idx in tqdm(range(len(test_data)), desc="Processing Test Samples"):
            current_emb = test_embeddings[test_idx].unsqueeze(0).to(self.device)

            # 第一阶段：按OOD分数筛选候选池
            top_ood_indices = torch.topk(-self.train_ood_scores, k=ood_number).indices
            candidate_embeddings = self.train_embeddings[top_ood_indices]

            # 第二阶段：计算所有候选的相似度并排序
            similarities = F.cosine_similarity(current_emb, candidate_embeddings)
            sorted_scores, sorted_indices = torch.sort(similarities, descending=True)

            # 先用相似度选出前2个样本
            selected_indices = [int(top_ood_indices[i.item()]) for i in sorted_indices[:2]]
            selected_embeddings = torch.stack([self.train_embeddings[idx] for idx in selected_indices])

            # 计算当前马氏距离矩阵
            if len(selected_embeddings) >= 2:
                try:
                    M_matrix = compute_mahalanobis_matrix_torch(selected_embeddings)
                    current_mean_maha = torch.mean(
                        M_matrix[torch.triu_indices(M_matrix.shape[0], M_matrix.shape[1], offset=1)]
                    )
                except:
                    current_mean_maha = float('inf')
            else:
                current_mean_maha = float('inf')

            # 从剩余候选中选择第三个样本
            remaining_indices = sorted_indices[2:]
            for i in remaining_indices:
                candidate_idx = int(top_ood_indices[i.item()])
                candidate_emb = self.train_embeddings[candidate_idx].unsqueeze(0)

                # 计算加入候选后的马氏距离
                new_embeddings = torch.cat([selected_embeddings, candidate_emb])
                try:
                    new_M_matrix = compute_mahalanobis_matrix_torch(new_embeddings)
                    new_mean_maha = torch.mean(
                        new_M_matrix[torch.triu_indices(new_M_matrix.shape[0], new_M_matrix.shape[1], offset=1)]
                    )

                    # 如果新样本能保持或增加多样性则选择
                    if new_mean_maha >= current_mean_maha:
                        selected_indices.append(candidate_idx)
                        break
                except:
                    continue

            # if len(selected_indices) < 3 and len(remaining_indices) > 0:
            #     selected_indices.append(int(top_ood_indices[remaining_indices[0].item()]))

            rtr_idx_list.append(selected_indices[:3])

        return rtr_idx_list


if __name__ == "__main__":
    retriever = OODRetriever()
    ood_numbers = [800]
    for ood_number in ood_numbers:
        rtr_idx_list = retriever.get_rtr_idx_list(ood_number=ood_number)
        print(rtr_idx_list)
        output_file = f'/ener/gemma_mashi_sim3_ood{ood_number}.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(rtr_idx_list, f, ensure_ascii=False, indent=4)
        print(f" OOD数量 {ood_number} 的结果已保存到 {output_file}")

