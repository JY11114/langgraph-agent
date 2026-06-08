"""
Tool-Calling Agent（短期 + 长期记忆）

短期记忆：MemorySaver checkpointer，维护单次会话上下文
长期记忆：ChromaDB 持久化存储，跨会话保留关键结论；
         会话结束后自动摘要写入，下次启动时语义检索注入
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
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
_checkpointer = MemorySaver()

_agent = create_agent(
    llm,
    tools,
    system_prompt="""你是一位专业的金融分析师助手，拥有以下工具：
- search_knowledge_base：检索本地金融研报（宁德时代/贵州茅台/比亚迪/新能源行业）
- web_search：搜索互联网实时市场动态
- get_stock_price：查询 A 股实时股价
- calculate_financial_metrics：计算 PE 估值、增长率、市值等金融指标
- save_to_memory：将重要结论保存到长期记忆
- recall_from_memory：检索历史会话中的关键信息

优先使用研报知识库获取专业数据，数值计算请调用计算工具。
回答时不要提及"长期记忆"、"历史记录"等内部机制，直接给出分析结论即可。""",
    checkpointer=_checkpointer,
)


def chat(question: str, session_id: str = "default") -> str:
    # 1. 检索长期记忆，拼入用户消息
    memories = _long_term_memory.recall(question, top_k=3)
    if memories:
        ctx = "\n".join(f"  · {m}" for m in memories)
        input_text = f"【长期记忆参考】\n{ctx}\n\n【用户问题】{question}"
    else:
        input_text = question

    config = {"configurable": {"thread_id": session_id}}
    result = _agent.invoke({"messages": [HumanMessage(content=input_text)]}, config=config)

    # 取最后一条 AI 消息
    messages = result.get("messages", [])
    output = messages[-1].content if messages else ""
    if isinstance(output, list):
        output = output[0].get("text", str(output))

    # 2. 会话满 4 条后自动摘要写入长期记忆
    state = _agent.get_state(config)
    history_messages = state.values.get("messages", [])
    if len(history_messages) >= 4:
        history_dicts = [
            {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in history_messages
        ]
        _long_term_memory.summarize_and_save(session_id, history_dicts, _llm_client)

    return str(output)


# ── CLI 入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    session_id = "default"
    print("金融研报问答 Agent")
    print("输入 exit 或按 Ctrl+C 退出")
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
        answer = chat(q, session_id=session_id)
        print(f"\n助手：{answer}")
        print("-" * 60)
