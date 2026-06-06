"""
LangGraph 状态图示例：搜索 → 质量判断 → 分析

演示 LangGraph 核心概念：StateGraph、条件边（conditional_edges）、重试机制。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from core.config import Config
from service.rag_service import RAGService

llm = ChatAnthropic(model=Config.MODEL_NAME, api_key=Config.ANTHROPIC_API_KEY)
_rag = RAGService()


# ── 状态定义 ──────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    search_result: str
    final_answer: str
    retry_count: int


# ── 节点函数 ──────────────────────────────────────────────────
def search_node(state: AgentState) -> AgentState:
    """调用 RAG 工具检索研报知识库。"""
    print("[搜索节点] 检索本地研报知识库...")
    result = _rag.chat(state["question"])
    state["search_result"] = result
    print("[搜索节点] 检索完成")
    return state


def analysis_node(state: AgentState) -> AgentState:
    """基于检索结果生成专业分析。"""
    print("[分析节点] 生成最终分析...")
    response = llm.invoke(
        f"你是一位资深金融分析师。请基于以下研报信息对问题给出专业分析。\n\n"
        f"研报信息：{state['search_result']}\n\n"
        f"用户问题：{state['question']}\n\n"
        f"请给出核心结论和数据支撑。"
    )
    state["final_answer"] = response.content
    state["retry_count"] += 1
    print("[分析节点] 完成")
    return state


def should_retry(state: AgentState) -> Literal["search", "analysis"]:
    """条件边：检索结果不足则重试，最多 3 次。"""
    if state["retry_count"] >= 3:
        print("已达最大重试次数，强制继续分析")
        return "analysis"
    if len(state["search_result"]) < 50:
        print(f"搜索结果不足，重试（第 {state['retry_count'] + 1} 次）")
        return "search"
    print("搜索结果充分，进入分析")
    return "analysis"


# ── 构建图 ────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("search", search_node)
graph.add_node("analysis", analysis_node)
graph.set_entry_point("search")
graph.add_conditional_edges(
    "search",
    should_retry,
    {"search": "search", "analysis": "analysis"},
)
graph.add_edge("analysis", END)

app = graph.compile()


# ── 入口 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "宁德时代值得投资吗",
        "贵州茅台的核心竞争优势是什么",
        "新能源行业2025年的投资机会在哪里",
    ]

    for q in questions:
        print("\n" + "=" * 50)
        print(f"问题：{q}")
        print("=" * 50)
        result = app.invoke({
            "question": q,
            "search_result": "",
            "final_answer": "",
            "retry_count": 0,
        })
        print("\n最终答案：")
        print(result["final_answer"])
