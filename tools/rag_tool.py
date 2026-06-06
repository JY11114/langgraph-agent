from langchain.tools import tool
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from service.rag_service import RAGService

_rag = None


def _get_rag() -> RAGService:
    global _rag
    if _rag is None:
        _rag = RAGService()
    return _rag


@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索本地金融研报知识库。
    当需要查询券商研报、行业分析、公司深度研究、投资建议等专业内容时使用。
    知识库包含：宁德时代、贵州茅台、比亚迪的深度研报，以及新能源行业策略报告。
    """
    try:
        return _get_rag().chat(query)
    except Exception as e:
        return f"知识库查询失败：{str(e)}"
