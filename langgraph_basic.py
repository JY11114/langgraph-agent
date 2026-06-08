"""
LangGraph 状态图 — 任务规划与执行闭环

执行流程：
  search_node（RAG 检索）
      │
  should_retry（条件边）
      ├── 结果不足 → 重新检索（最多 3 次）
      └── 充分    → analysis_node（LLM 综合分析）
                        │
                       END

体现简历中的「规划—工具调用—反思—重试」闭环。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict
from langgraph.graph import StateGraph, END
from core.config import Config
from core.llm_client import LLMClient
from core.rag_engine import RAGEngine

# ── 状态定义 ──────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str        # 原始用户问题
    search_results: list[str]  # 每次检索返回的片段列表
    retry_count: int     # 已重试次数
    analysis: str        # 最终分析结果


MAX_RETRIES = 3
MIN_RESULTS = 2          # 认为"结果充分"的最少片段数

# ── 共享实例 ──────────────────────────────────────────────────
_llm = LLMClient()
_rag_engine: RAGEngine | None = None


def _get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
        _rag_engine.build_vector_db()
    return _rag_engine


# ── 节点 ──────────────────────────────────────────────────────
def search_node(state: AgentState) -> AgentState:
    """RAG 检索节点：对问题做查询改写后执行混合检索+精排。"""
    engine = _get_rag_engine()
    rewritten = engine.rewrite_query(state["question"], _llm)
    candidates = engine.hybrid_search(rewritten)
    docs = engine.rerank(rewritten, candidates, top_k=5)

    print(f"  [search_node] 第 {state['retry_count'] + 1} 次检索，召回 {len(docs)} 条片段")
    return {
        **state,
        "search_results": docs,
        "retry_count": state["retry_count"] + 1,
    }


def analysis_node(state: AgentState) -> AgentState:
    """分析节点：将检索结果交给 LLM 综合推理，生成最终回答。"""
    context = "\n\n".join(state["search_results"])
    messages = [
        {
            "role": "user",
            "content": f"参考资料：\n{context}\n\n请基于以上研报内容回答：{state['question']}",
        }
    ]
    answer = _llm.send_message(
        messages=messages,
        system=(
            "你是一位专业的金融分析师助手，请仅依据给定参考资料作答，"
            "标注关键数据来源，无依据时明确说明，不要编造内容。"
        ),
    )
    print(f"  [analysis_node] 分析完成")
    return {**state, "analysis": answer}


# ── 条件边（反思与重试逻辑）─────────────────────────────────
def should_retry(state: AgentState) -> str:
    """
    判断检索结果是否充分：
      - 片段数不足且未超过最大重试次数 → retry（重新检索）
      - 否则 → analyze（进入分析节点）
    """
    if len(state["search_results"]) < MIN_RESULTS and state["retry_count"] < MAX_RETRIES:
        print(f"  [should_retry] 结果不足（{len(state['search_results'])} 条），触发重试 "
              f"（{state['retry_count']}/{MAX_RETRIES}）")
        return "retry"
    return "analyze"


# ── 构建状态图 ────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("search", search_node)
    graph.add_node("analysis", analysis_node)

    graph.set_entry_point("search")

    graph.add_conditional_edges(
        "search",
        should_retry,
        {"retry": "search", "analyze": "analysis"},
    )

    graph.add_edge("analysis", END)

    return graph.compile()


# ── 对外接口 ──────────────────────────────────────────────────
def run(question: str) -> str:
    """执行完整的检索→反思→分析闭环，返回最终分析结果。"""
    app = build_graph()
    initial_state: AgentState = {
        "question": question,
        "search_results": [],
        "retry_count": 0,
        "analysis": "",
    }
    final_state = app.invoke(initial_state)
    return final_state["analysis"]


# ── CLI 入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    print(" 金融研报问答")
    print("输入 exit 退出")
    print("=" * 60)

    while True:
        try:
            q = input("\n你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出。")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出"):
            print("已退出。")
            break

        print()
        answer = run(q)
        print(f"\n助手：{answer}")
        print("-" * 60)
