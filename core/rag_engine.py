import os
import jieba
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from core.config import Config


class RAGEngine:
    def __init__(self):
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )
        self.chunks: list[str] = []
        self.bm25 = None

        print("正在加载 Reranker 模型...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        print("RAG 引擎初始化完成")

    def load_knowledge(self) -> list[str]:
        """加载 knowledge_base/ 目录下所有 .txt 文件。"""
        kb_dir = Config.KNOWLEDGE_BASE_DIR
        all_chunks: list[str] = []

        for filename in sorted(os.listdir(kb_dir)):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = self.splitter.split_text(content)
            all_chunks.extend(chunks)
            print(f"  {filename} → {len(chunks)} chunks")

        self.chunks = all_chunks
        print(f"知识库加载完成，共 {len(all_chunks)} 个 chunks")
        return all_chunks

    def build_vector_db(self):
        chunks = self.load_knowledge()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadata_list = [{"chunk_index": i} for i in range(len(chunks))]

        self.collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=metadata_list,
        )

        tokenized = [list(jieba.cut(c)) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        print(f"向量数据库 + BM25 索引构建完成，共 {len(chunks)} 条")

    def search(self, query: str, top_k: int = None) -> list[str]:
        if top_k is None:
            top_k = Config.TOP_K
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = results["documents"][0]
        print(f"向量检索：找到 {len(docs)} 条")
        return docs

    def bm25_search(self, query: str, top_k: int = None) -> list[str]:
        if top_k is None:
            top_k = Config.TOP_K
        if not self.bm25:
            return []
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        docs = [self.chunks[i] for i in top_idx]
        print(f"BM25 检索：找到 {len(docs)} 条")
        return docs

    def hybrid_search(self, query: str, top_k: int = None) -> list[str]:
        if top_k is None:
            top_k = Config.TOP_K * 4

        vector_results = self.search(query, top_k=top_k)
        bm25_results = self.bm25_search(query, top_k=top_k)

        rrf_scores: dict[str, float] = {}
        k = 60
        for rank, doc in enumerate(vector_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (k + rank + 1)
        for rank, doc in enumerate(bm25_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (k + rank + 1)

        sorted_docs = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)
        result = sorted_docs[:top_k]
        print(f"混合检索（RRF）：{len(result)} 条候选")
        return result

    def rewrite_query(self, query: str, llm) -> str:
        prompt = f"""将以下用户问题改写为更适合在金融研报知识库中检索的形式。
要求：补全省略的公司名、去除口语化表达、保持原意、只返回改写后的问题。

用户问题：{query}"""
        messages = [{"role": "user", "content": prompt}]
        rewritten = llm.send_message(messages=messages)
        print(f"Query 改写：{query} → {rewritten}")
        return rewritten

    def rerank(self, query: str, docs: list[str], top_k: int = 3) -> list[str]:
        if not docs:
            return []
        pairs = [(query, doc) for doc in docs]
        scores = self.reranker.predict(pairs)
        top_idx = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [docs[i] for i in top_idx]
