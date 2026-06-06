from core.llm_client import LLMClient
from core.rag_engine import RAGEngine
from core.config import Config


class RAGService:
    def __init__(self):
        self.llm = LLMClient()
        self.engine = RAGEngine()
        self.engine.build_vector_db()

    def chat(self, user_input: str) -> str:
        rewritten = self.engine.rewrite_query(user_input, self.llm)
        candidates = self.engine.hybrid_search(rewritten)
        docs = self.engine.rerank(rewritten, candidates, top_k=5)

        context = "\n\n".join(docs)
        messages = [{"role": "user", "content": f"参考资料:\n{context}\n\n用户问题:{user_input}"}]
        reply = self.llm.send_message(messages=messages, system=Config.RAG_SYSTEM_PROMPT)
        return reply
