"""
长期记忆模块

使用 ChromaDB 持久化存储跨会话的关键结论与用户偏好，
支持语义检索与会话结束后的自动摘要写入。
"""

import time
import chromadb
from chromadb.utils import embedding_functions


class LongTermMemory:
    def __init__(self, db_path: str = "./memory_db"):
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="agent_long_term_memory",
            embedding_function=self.embedding_fn,
        )

    def save(self, content: str, session_id: str = "default", category: str = "fact") -> str:
        """保存一条记忆，返回记忆 ID。"""
        memory_id = f"{session_id}_{int(time.time() * 1000)}"
        self.collection.upsert(
            documents=[content],
            ids=[memory_id],
            metadatas=[{
                "session_id": session_id,
                "timestamp": int(time.time()),
                "category": category,
            }],
        )
        return memory_id

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        """语义检索相关历史记忆。"""
        total = self.collection.count()
        if total == 0:
            return []
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, total),
            )
            return results["documents"][0] if results["documents"] else []
        except Exception:
            return []

    def summarize_and_save(self, session_id: str, history: list[dict], llm) -> str:
        """
        会话结束后提取关键结论，自动写入长期记忆。
        由 agent 在每轮结束时调用。
        """
        if len(history) < 2:
            return ""

        conversation = "\n".join(
            f"{'用户' if m['role'] == 'user' else 'AI'}: {m['content'][:300]}"
            for m in history[-10:]
        )

        prompt = f"""从以下对话中提取1-3条值得长期记住的金融分析结论或用户偏好，每条一行，简洁直接。
如果没有值得记忆的内容，只返回"无"。

对话：
{conversation}"""

        summary = llm.send_message(messages=[{"role": "user", "content": prompt}])

        if summary.strip() == "无":
            return ""

        for line in summary.strip().split("\n"):
            line = line.strip().lstrip("·-•1234567890. ")
            if len(line) > 5:
                self.save(line, session_id=session_id, category="session_summary")

        return summary

    def count(self) -> int:
        return self.collection.count()
