import re
import json
import numpy as np
import torch
import faiss
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel


def load_icl_data(icl_file_path):
    icl_data = []
    with open(icl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                icl_data.append(json.loads(line))
    return icl_data


class LocalEmbeddingModel:
    def __init__(self, model_path_or_name="shibing624/text2vec-base-chinese"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        self.model = AutoModel.from_pretrained(model_path_or_name).to(self.device)
        self.model.eval()
        print("本地嵌入模型加载完成")

    @torch.no_grad()
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])
        return np.array(embeddings)


class ComplexityAnalyzer:
    RELATION_KEYWORDS = [
        "整合", "不整合", "平行不整合", "角度不整合",
        "侵入接触", "沉积接触", "断层接触", "侵入", "错断", "切穿",
        "覆盖", "相交", "位于", "包含", 
        "东", "西", "南", "北", "东南", "东北", "西北", "西南",
        "顶部", "底部", "上覆", "下伏"
    ]

    def __init__(self, base_model_url="http://0.0.0.0:8000/v1",
                 model_name="本地模型"):
        self.client = OpenAI(api_key="0", base_url=base_model_url)
        self.model_name = model_name
        pattern = "|".join(re.escape(kw) for kw in self.RELATION_KEYWORDS)
        self.relation_pattern = re.compile(pattern)

    def zero_shot_entity_count(self, text):
        try:
            prompt = (f"请从以下地质文本中提取所有地质实体（包括岩石、地层、地质构造、地名），"
                      f"仅返回实体列表，用逗号分隔，不要包含任何其他文字。\n文本：{text}\n实体列表：")
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name, temperature=0.0, max_tokens=256
            )
            content = response.choices[0].message.content.strip()
            entities = [e.strip() for e in content.replace("，", ",").split(",") if e.strip()]
            return max(len(entities), 1)
        except Exception as e:
            print(f"[Warning] 零样本实体抽取失败: {e}, 回退默认值1")
            return 1

    def keyword_relation_count(self, text):
        matches = self.relation_pattern.findall(text)
        return max(len(matches), 1)

    def calculate_struct_score(self, text):
        n_ent = self.zero_shot_entity_count(text)
        n_rel = self.keyword_relation_count(text)
        score = n_rel / n_ent
        return score, n_ent, n_rel

    @staticmethod
    def classify_complexity(score):
        if score <= 0.5 + 1e-9:
            return "normal"
        elif score < 1.0:
            return "seo"
        else:
            return "epo"


class AdaptiveRetriever:
    def __init__(self, icl_file_path, embedding_model_path=None,
                 base_model_url="http://0.0.0.0:8000/v1",
                 llm_model_name="本地模型"):
        self.icl_data = load_icl_data(icl_file_path)
        self.embedding_model = LocalEmbeddingModel(embedding_model_path)
        self.analyzer = ComplexityAnalyzer(base_model_url=base_model_url, model_name=llm_model_name)

        # 预计算嵌入 & FAISS
        print("正在预计算ICL嵌入向量...")
        self.icl_texts = [item['text'] for item in self.icl_data]
        self.icl_embeddings = self.embedding_model.encode(self.icl_texts)
        norm = np.linalg.norm(self.icl_embeddings, axis=1, keepdims=True)
        self.icl_embeddings_normalized = self.icl_embeddings / norm
        self.d = self.icl_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(np.float32(self.icl_embeddings_normalized))

        # 预计算ICL复杂度得分
        print("正在预计算ICL结构复杂度得分...")
        self.icl_complexity_scores = np.array([
            self.analyzer.calculate_struct_score(t)[0]
            for t in tqdm(self.icl_texts, desc="ICL复杂度")
        ])
        print(f"AdaptiveRetriever 初始化完成，共 {len(self.icl_data)} 条示例")

    def adaptive_retrieve(self, input_text, top_n=3, semantic_weight=0.5, struct_weight=0.5):

        input_score, n_ent, n_rel = self.analyzer.calculate_struct_score(input_text)
        ctype = ComplexityAnalyzer.classify_complexity(input_score)
        print(f"[Adaptive] S_struct={input_score:.3f}, type={ctype}, N_ent={n_ent}, N_rel={n_rel}")

        # 语义相似度搜索
        emb = self.embedding_model.encode([input_text])
        emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        sem_scores, sem_indices = self.index.search(np.float32(emb_norm), len(self.icl_data))

        if ctype == "normal":
            print("[Adaptive] 策略: 纯语义检索")
            results = []
            for idx, score in zip(sem_indices[0][:top_n], sem_scores[0][:top_n]):
                results.append({
                    'text': self.icl_data[idx]['text'],
                    'spo_list': self.icl_data[idx]['spo_list'],
                    'combined_score': float(score),
                    'semantic_score': float(score),
                    'struct_score': float(self.icl_complexity_scores[idx]),
                    'complexity_type': ctype,
                    'input_struct_score': float(input_score)
                })
            return results
        else:
            print(f"[Adaptive] 策略: 混合检索 (sem_w={semantic_weight}, struct_w={struct_weight})")
            struct_sim = np.clip(1.0 - np.abs(self.icl_complexity_scores - input_score), 0, 1)
            s_flat = sem_scores[0]
            s_min, s_max = s_flat.min(), s_flat.max()
            norm_sem = (s_flat - s_min) / (s_max - s_min) if (s_max - s_min) > 1e-9 else np.ones_like(s_flat)
            combined = semantic_weight * norm_sem + struct_weight * struct_sim
            top_idx = np.argsort(combined)[::-1][:top_n]

            results = []
            for idx in top_idx:
                pos = np.where(sem_indices[0] == idx)[0]
                raw_sem = float(sem_scores[0][pos[0]]) if len(pos) > 0 else 0.0
                results.append({
                    'text': self.icl_data[idx]['text'],
                    'spo_list': self.icl_data[idx]['spo_list'],
                    'combined_score': float(combined[idx]),
                    'semantic_score': raw_sem,
                    'struct_score': float(self.icl_complexity_scores[idx]),
                    'struct_similarity': float(struct_sim[idx]),
                    'complexity_type': ctype,
                    'input_struct_score': float(input_score)
                })
            return results