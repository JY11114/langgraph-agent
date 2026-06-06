import sys
import os
from langchain.tools import tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory.long_term_memory import LongTermMemory

_memory = LongTermMemory()


@tool
def save_to_memory(content: str) -> str:
    """
    将重要结论或信息保存到长期记忆，供未来会话检索。
    适用场景：用户明确要求记住某事、对话产生重要投资结论、识别到用户偏好时。
    """
    memory_id = _memory.save(content)
    return f"已保存到长期记忆：{content}（ID: {memory_id[:20]}）"


@tool
def recall_from_memory(query: str) -> str:
    """
    从长期记忆中检索与查询相关的历史信息。
    适用场景：用户询问之前讨论过的内容、需要跨会话上下文时。
    """
    memories = _memory.recall(query, top_k=3)
    if not memories:
        return "长期记忆中暂无相关记录。"
    result = "\n".join(f"  · {m}" for m in memories)
    return f"【长期记忆】\n{result}"
