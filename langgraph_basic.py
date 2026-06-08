
import re
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict
from langgraph.graph import StateGraph, END
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

from core.config import Config
from core.llm_client import LLMClient
from core.rag_engine import RAGEngine
from memory.long_term_memory import LongTermMemory


# CrewAI 兼容的 DuckDuckGo 工具（不需要 API Key）
class _DuckDuckGoTool(BaseTool):
    name: str = "web_search"
    description: str = "搜索互联网上的实时信息，用于查询最新新闻、市场动态、机构评级。"

    def _run(self, query: str) -> str:
        return DuckDuckGoSearchRun().run(query)

_ddg_tool = _DuckDuckGoTool()


MAX_RETRIES = 3        # RAG 搜索最大重试次数
MAX_ANALYSIS_RETRY = 2 # 分析最大重试次数
MIN_RAG_RESULTS = 2    # 认为 RAG 结果充分的最少片段数
MIN_WEB_LEN = 100      # 认为联网结果充分的最少字数
QUALITY_PASS = 6       # 质量评分及格线（1-10）


class AgentState(TypedDict):
    question: str
    rag_results: list[str]     # 知识库检索片段
    web_results: str           # 联网搜索摘要
    past_memories: list[str]   # 长期记忆召回结果
    retry_count: int           # RAG 搜索已重试次数
    analysis: str              # 综合分析结果
    analysis_retry: int        # 分析已重试次数
    quality_score: int         # 质量评分


_llm = LLMClient()
_rag_engine: RAGEngine | None = None
_memory = LongTermMemory()
# CrewAI 通过 litellm 调用 Anthropic，模型名加 anthropic/ 前缀
_CREW_MODEL = f"anthropic/{Config.MODEL_NAME}"


def _get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
        _rag_engine.build_vector_db()
    return _rag_engine


def rag_search_node(state: AgentState) -> AgentState:
    """知识库 RAG 混合检索（向量 + BM25 + Reranker）。"""
    engine = _get_rag_engine()
    rewritten = engine.rewrite_query(state["question"], _llm)
    candidates = engine.hybrid_search(rewritten)
    docs = engine.rerank(rewritten, candidates, top_k=5)
    print(f"  [rag_search_node] 第 {state['retry_count'] + 1} 次检索，召回 {len(docs)} 条片段")
    return {
        **state,
        "rag_results": docs,
        "retry_count": state["retry_count"] + 1,
    }


def web_search_node(state: AgentState) -> AgentState:
    """CrewAI Agent + DuckDuckGo 联网搜索节点（无需 API Key）。"""
    researcher = Agent(
        role="金融信息研究员",
        goal="从网络上搜集投资标的的最新资讯、新闻和市场数据",
        backstory=(
            "你是专注于 A 股和新能源领域的金融信息研究员，"
            "善于从网络上找到有价值的实时信息并提炼关键内容。"
        ),
        tools=[_ddg_tool],
        llm=_CREW_MODEL,
        verbose=False,
    )
    task = Task(
        description=(
            f"请搜索关于「{state['question']}」的最新信息，"
            f"重点关注近期新闻、市场动态、机构评级，整理成简洁摘要。"
        ),
        agent=researcher,
        expected_output="300 字以内的信息摘要，包含关键数据和来源说明",
    )
    crew = Crew(agents=[researcher], tasks=[task], verbose=False)
    result = crew.kickoff()
    web_text = str(result)
    print(f"  [web_search_node] 联网搜索完成，字数={len(web_text)}")
    return {**state, "web_results": web_text}


def analysis_node(state: AgentState) -> AgentState:
    """综合 RAG 研报 + 联网资讯 + 历史记忆，给出投资分析结论。"""
    # 召回与本次问题相关的历史记忆
    past = _memory.recall(state["question"], top_k=3)
    memory_block = ""
    if past:
        memory_block = "【历史分析记忆】\n" + "\n".join(f"- {m}" for m in past) + "\n\n"
        print(f"  [analysis_node] 召回 {len(past)} 条历史记忆")

    rag_context = "\n\n".join(state["rag_results"])
    messages = [
        {
            "role": "user",
            "content": (
                f"{memory_block}"
                f"【知识库研报摘录】\n{rag_context}\n\n"
                f"【网络实时资讯】\n{state['web_results']}\n\n"
                f"请结合以上所有信息，对「{state['question']}」给出专业分析结论，"
                f"注明哪些结论来自研报、哪些来自实时资讯。"
            ),
        }
    ]
    answer = _llm.send_message(
        messages=messages,
        system=(
            "你是一位专业金融分析师，综合研报和实时资讯进行分析，"
            "区分研报数据与网络信息，无依据时明确说明，不要编造内容。"
        ),
    )
    print(f"  [analysis_node] 第 {state['analysis_retry'] + 1} 次分析完成")
    return {
        **state,
        "past_memories": past,
        "analysis": answer,
        "analysis_retry": state["analysis_retry"] + 1,
        "quality_score": 0,
    }


def save_memory_node(state: AgentState) -> AgentState:
    """分析达标后，提炼关键结论写入长期记忆。"""
    history = [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": state["analysis"]},
    ]
    summary = _memory.summarize_and_save(
        session_id=f"session_{int(__import__('time').time())}",
        history=history,
        llm=_llm,
    )
    if summary and summary.strip() != "无":
        print(f"  [save_memory_node] 已写入长期记忆（共 {_memory.count()} 条）")
    else:
        print(f"  [save_memory_node] 无值得记录的内容")
    return state


def quality_check_node(state: AgentState) -> AgentState:
    """LLM 对分析结果打分（1-10），评估逻辑、数据、结论质量。"""
    messages = [
        {
            "role": "user",
            "content": (
                f"请对以下金融分析回答打分（1-10 分），标准：逻辑清晰、有数据支撑、结论明确。\n"
                f"只返回一个整数，不要其他文字。\n\n"
                f"问题：{state['question']}\n"
                f"回答：{state['analysis']}"
            ),
        }
    ]
    score_str = _llm.send_message(messages=messages)
    match = re.search(r'\b(10|[1-9])\b', score_str)
    score = int(match.group()) if match else 5
    print(f"  [quality_check_node] 质量评分 {score}/10")
    return {**state, "quality_score": score}


def should_retry(state: AgentState) -> str:
    rag_ok = len(state["rag_results"]) >= MIN_RAG_RESULTS
    web_ok = len(state["web_results"]) >= MIN_WEB_LEN
    if (not rag_ok or not web_ok) and state["retry_count"] < MAX_RETRIES:
        print(
            f"  [should_retry] 结果不足（RAG={len(state['rag_results'])}条 "
            f"Web={len(state['web_results'])}字），重试 {state['retry_count']}/{MAX_RETRIES}"
        )
        return "retry"
    return "analyze"


def should_redo(state: AgentState) -> str:
    if state["quality_score"] < QUALITY_PASS and state["analysis_retry"] < MAX_ANALYSIS_RETRY:
        print(
            f"  [should_redo] 质量不足（{state['quality_score']} 分），"
            f"重新分析 {state['analysis_retry']}/{MAX_ANALYSIS_RETRY}"
        )
        return "redo"
    print(f"  [should_redo] 质量达标（{state['quality_score']} 分），完成")
    return "done"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("rag_search", rag_search_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("rag_search")
    graph.add_edge("rag_search", "web_search")

    graph.add_conditional_edges(
        "web_search",
        should_retry,
        {"retry": "rag_search", "analyze": "analysis"},
    )

    graph.add_edge("analysis", "quality_check")

    graph.add_conditional_edges(
        "quality_check",
        should_redo,
        {"redo": "analysis", "done": "save_memory"},
    )

    graph.add_edge("save_memory", END)

    return graph.compile()


def run(question: str) -> str:
    app = build_graph()
    initial_state: AgentState = {
        "question": question,
        "rag_results": [],
        "web_results": "",
        "past_memories": [],
        "retry_count": 0,
        "analysis": "",
        "analysis_retry": 0,
        "quality_score": 0,
    }
    last_state = None
    for step in app.stream(initial_state):
        node_name = list(step.keys())[0]
        state = step[node_name]
        print(f"\n>>> [LangGraph] 节点: {node_name}  retry={state.get('retry_count', '-')}")
        last_state = state
    return last_state["analysis"]


if __name__ == "__main__":
    print("金融研报Agent")
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
