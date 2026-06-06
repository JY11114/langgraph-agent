"""
Tool-Calling Agent（短期 + 长期记忆）

短期记忆：RunnableWithMessageHistory，维护单次会话上下文
长期记忆：ChromaDB 持久化存储，跨会话保留关键结论；
         会话结束后自动摘要写入，下次启动时语义检索注入
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from core.config import Config
from core.llm_client import LLMClient
from memory.long_term_memory import LongTermMemory
from tools import (
    search_knowledge_base,
    web_search,
    get_stock_price,
    calculate_financial_metrics,
    save_to_memory,
    recall_from_memory,
)

tools = [
    search_knowledge_base,
    web_search,
    get_stock_price,
    calculate_financial_metrics,
    save_to_memory,
    recall_from_memory,
]

llm = ChatAnthropic(model=Config.MODEL_NAME, api_key=Config.ANTHROPIC_API_KEY)
_llm_client = LLMClient()
_long_term_memory = LongTermMemory()

# ── 会话历史存储 ──────────────────────────────────────────────
_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def _build_agent(long_term_context: str = ""):
    """构建 Agent，将长期记忆注入 System Prompt。"""
    memory_section = (
        f"\n\n【长期记忆】以下是与本次对话相关的历史记录，可作为背景参考：\n{long_term_context}"
        if long_term_context else ""
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是一位专业的金融分析师助手，拥有以下工具：
- search_knowledge_base：检索本地金融研报（宁德时代/贵州茅台/比亚迪/新能源行业）
- web_search：搜索互联网实时市场动态
- get_stock_price：查询 A 股实时股价
- calculate_financial_metrics：计算 PE 估值、增长率、市值等金融指标
- save_to_memory：将重要结论保存到长期记忆
- recall_from_memory：检索历史会话中的关键信息

优先使用研报知识库获取专业数据，数值计算请调用计算工具。{memory_section}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6)
    return RunnableWithMessageHistory(
        executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def chat(question: str, session_id: str = "default") -> str:
    """
    单轮对话入口。
    - 检索长期记忆注入上下文
    - 调用 Agent 回答
    - 自动将本轮会话摘要写入长期记忆
    """
    # 1. 检索长期记忆
    memories = _long_term_memory.recall(question, top_k=3)
    long_term_ctx = "\n".join(f"  · {m}" for m in memories) if memories else ""

    # 2. 构建带记忆上下文的 Agent
    agent_with_memory = _build_agent(long_term_ctx)
    config = {"configurable": {"session_id": session_id}}

    # 3. 执行
    result = agent_with_memory.invoke({"input": question}, config=config)
    output = result.get("output", "")
    if isinstance(output, list):
        output = output[0].get("text", str(output))

    # 4. 会话结束后自动摘要写入长期记忆
    history = get_session_history(session_id)
    if len(history.messages) >= 4:  # 至少 2 轮对话再摘要
        history_dicts = [
            {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in history.messages
        ]
        _long_term_memory.summarize_and_save(session_id, history_dicts, _llm_client)

    return str(output)


# ── CLI 入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    session_id = "demo"
    print(f"金融研报问答 Agent（长期记忆已存储 {_long_term_memory.count()} 条）")
    print("=" * 60)

    demo_questions = [
        "宁德时代2024年Q3毛利率是多少？储能业务增速如何？",
        "基于研报数据，帮我算一下宁德时代按2025年EPS=12.8、PE=16.5估算的目标价",
        "贵州茅台现在股价多少？相比研报目标价还有多少空间？",
        "把刚才宁德时代的估值结论保存到记忆里",
    ]

    for q in demo_questions:
        print(f"\n用户：{q}")
        answer = chat(q, session_id=session_id)
        print(f"助手：{answer}")
        print("-" * 60)

    print(f"\n会话结束，长期记忆现有 {_long_term_memory.count()} 条")
