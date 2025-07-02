import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BertTokenizer, BertModel
from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger
from typing import List, Optional
from accelerate import Accelerator
from tqdm import trange
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


class ColBERTRetriever(BaseRetriever):
    """基于BERT的In-context Learning Retriever"""

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 accelerator: Optional[Accelerator] = None,
                 model_name: Optional[str] = '/bert-base-uncased') -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)

        # # 加载 BERT 模型和 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print("eval")
        self.model.eval()


        # 初始化索引和测试语料
        self.index_corpus = self._embed_corpus(self.index_ds)
        self.test_corpus = [self.tokenizer.tokenize(data) for data in
                            self.dataset_reader.generate_input_field_corpus(self.test_ds)]
        # print("index_corpus shape:", len(self.index_corpus),type(self.index_corpus),self.index_corpus)
        # print("test_corpus shape:", len(self.test_corpus),type(self.index_corpus))

    def _get_embedding(self, text: str) -> torch.Tensor:
        """生成单条文本的嵌入"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(
            self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        return output.last_hidden_state.mean(dim=1)  # Mean pooling to get a single vector

    def _compute_cosine_similarity(self, query_embedding: torch.Tensor, index_embeddings: torch.Tensor) -> torch.Tensor:
        """计算查询与索引之间的余弦相似度"""
        # 归一化查询嵌入和索引嵌入
        query_embedding = query_embedding / query_embedding.norm(p=2, dim=-1, keepdim=True)  # 归一化查询嵌入
        index_embeddings = index_embeddings / index_embeddings.norm(p=2, dim=-1, keepdim=True)  # 归一化索引嵌入

        # 检查形状，确保是正确的
        # print("Query embedding shape:", query_embedding.shape)
        # print("Index embeddings shape:", index_embeddings.shape)

        query_embedding = query_embedding.squeeze(0)  # Shape: [768]
        index_embeddings = index_embeddings.squeeze(1)  # Shape: [n, 768]

        # 计算余弦相似度
        similarity_scores = torch.matmul(query_embedding, index_embeddings.T)  # [1, N]
        return similarity_scores

    def _embed_corpus(self, dataset) -> dict:
        """生成按标签分组的嵌入字典"""
        # print(dataset)
        label_embeddings = {}
        label_indices = {}
        # print(type(label_indices),len(label_indices))
        # print(type(label_embeddings),len(label_embeddings))
        for idx, item in enumerate(dataset):
            text = " ".join([str(item[col]) for col in self.dataset_reader.input_columns])
            # Premise = item['Premise']
            # # # .split(" ")
            # # # Premise = " ".join(Premise)
            # Hypothesis = item['Hypothesis']
            # # .split(" ")
            # # Hypothesis = " ".join(Hypothesis)
            # # text = Premise+ Hypothesis
            # text = Premise + Hypothesis
            # print(text)
            text = item['Text']
            label = item['Label']
            embedding = self._get_embedding(text)
            if label not in label_embeddings:
                label_embeddings[label] = []
                label_indices[label] = []
            label_embeddings[label].append(embedding)
            label_indices[label].append(idx)

        # 将每个标签的嵌入堆叠为张量
        for label in label_embeddings:
            label_embeddings[label] = torch.stack(label_embeddings[label])
        # print(len(label_embeddings), len(label_indices))
        return {'embeddings': label_embeddings, 'indices': label_indices}

# 使用相似度检索
    def retrieve(self) -> List[List]:
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
            test_item = self.test_ds[idx]
            # test_premise = test_item['Premise']
            # # test_premise = " ".join(test_premise)
            # test_hypothesis = test_item['Hypothesis']
            # # test_hypothesis = " ".join(test_hypothesis)
            # #

            text = test_item['Text']
            query = text
            # query = test_premise + test_hypothesis
            # query = " ".join(self.test_corpus[idx])
            query_embedding = self._get_embedding(query)

            # 按标签分组检索
            all_scores, all_indices = [], []
            for label in self.index_corpus['embeddings']:
                label_emb = self.index_corpus['embeddings'][label]
                similarity = self._compute_cosine_similarity(query_embedding, label_emb)
                # print(similarity)

                local_indices = similarity.topk(min(int(self.ice_num/3), len(label_emb))).indices
                global_indices = [self.index_corpus['indices'][label][i] for i in local_indices]

                all_scores.extend(similarity[local_indices].tolist())
                all_indices.extend(global_indices)

            sorted_pairs = sorted(zip(all_scores, all_indices), reverse=True, key=lambda x: x[0])
            # print(sorted_pairs)
            final_indices = [idx for _, idx in sorted_pairs[:self.ice_num]]
            rtr_idx_list.append(final_indices)

        return rtr_idx_list

#随机+采样
    # def retrieve(self) -> List[List]:
    #     rtr_idx_list = []
    #     logger.info("Randomly selecting data for test set...")
    #
    #     # 遍历测试集
    #     for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
    #         # 按标签分组随机选择
    #         all_indices = []
    #         for label in self.index_corpus['embeddings']:
    #             label_indices = self.index_corpus['indices'][label]
    #
    #             # 随机选择 ice_num 个索引（如果该标签下的索引数足够）
    #             random_indices = random.sample(label_indices, min(int(self.ice_num/3), len(label_indices)))
    #             all_indices.extend(random_indices)
    #
    #         # 打乱选择的索引，保证返回结果的随机性
    #         random.shuffle(all_indices)
    #
    #         # 只选择最相关的 ice_num 个
    #         final_indices = all_indices[:self.ice_num]
    #         rtr_idx_list.append(final_indices)
    #
    #     return rtr_idx_list

#解释的相似度
    # def retrieve(self) -> List[List]:
    #     rtr_idx_list = []
    #     logger.info("Retrieving data based on explanation similarity...")
    #
    #     # 遍历测试集
    #     for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
    #         # 获取查询的 explanation
    #         explanation = self.test_ds[idx]['explanation']
    #
    #         # 获取该查询的 explanation 嵌入
    #         query_embedding = self._get_embedding(explanation)
    #
    #         # 按标签分组检索
    #         all_scores, all_indices = [], []
    #         for label in self.index_corpus['embeddings']:
    #             label_emb = self.index_corpus['embeddings'][label]
    #             similarity = self._compute_cosine_similarity(query_embedding, label_emb)
    #
    #             print(similarity)
    #
    #             # 按照相似度取 top-k 个索引
    #             local_indices = similarity.topk(min(int(self.ice_num / 3), len(label_emb))).indices
    #             global_indices = [self.index_corpus['indices'][label][i] for i in local_indices]
    #
    #             all_scores.extend(similarity[local_indices].tolist())
    #             all_indices.extend(global_indices)
    #
    #         # 将所有相似度得分和索引按相似度降序排序
    #         sorted_pairs = sorted(zip(all_scores, all_indices), reverse=True, key=lambda x: x[0])
    #
    #         # 取前 ice_num 个最相关的索引
    #         final_indices = [idx for _, idx in sorted_pairs[:self.ice_num]]
    #         rtr_idx_list.append(final_indices)
    #
    #     return rtr_idx_list


