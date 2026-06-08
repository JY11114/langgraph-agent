"""
CrewAI 多 Agent 协作系统

角色分工：
  researcher  — 网络搜索，获取实时市场动态
  rag_analyst — 研报知识库检索，提供专业研究观点
  advisor     — 综合两路输出，生成投资分析报告

顺序流程：researcher → rag_analyst → advisor（context 共享）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool as crewai_tool
from langchain_anthropic import ChatAnthropic
from core.config import Config
from service.rag_service import RAGService
from langchain_community.tools import DuckDuckGoSearchRun

# ── 工具 ──────────────────────────────────────────────────────
_search = DuckDuckGoSearchRun()
_rag: RAGService | None = None


def _get_rag() -> RAGService:
    global _rag
    if _rag is None:
        _rag = RAGService()
    return _rag


@crewai_tool("网络搜索")
def web_search(query: str) -> str:
    """搜索互联网实时信息，适用于新闻、市场动态、最新价格等内容。"""
    try:
        return _search.run(query)
    except Exception as e:
        return f"搜索失败：{e}"


@crewai_tool("研报知识库检索")
def search_knowledge_base(query: str) -> str:
    """
    检索本地金融研报知识库，获取券商专业分析观点。
    知识库涵盖：宁德时代、贵州茅台、比亚迪深度研报及新能源行业策略报告。
    """
    try:
        return _get_rag().chat(query)
    except Exception as e:
        return f"知识库查询失败：{e}"


# ── LLM ──────────────────────────────────────────────────────
llm = ChatAnthropic(
    model=Config.MODEL_NAME,
    api_key=Config.ANTHROPIC_API_KEY,
    max_tokens=Config.MAX_TOKENS,
)

# ── Agents ────────────────────────────────────────────────────
researcher = Agent(
    role="市场研究员",
    goal="通过网络搜索获取目标公司或行业的最新市场动态、新闻及实时数据",
    backstory=(
        "你是一名专注 A 股市场的资深研究员，擅长通过互联网快速收集"
        "最新市场信息、政策动向与行业动态，为后续专业分析提供实时数据支撑。"
    ),
    tools=[web_search],
    llm=llm,
    verbose=True,
)

rag_analyst = Agent(
    role="研报分析师",
    goal="从本地研报知识库中检索专业分析观点，结合券商数据给出深度研究结论",
    backstory=(
        "你是一名专业的卖方研究分析师，深度研究过宁德时代、贵州茅台、比亚迪等"
        "头部公司及新能源行业，擅长从机构研报中提取核心投资逻辑与估值依据。"
    ),
    tools=[search_knowledge_base],
    llm=llm,
    verbose=True,
)

advisor = Agent(
    role="投资顾问",
    goal="综合市场实时信息与研报专业观点，输出结构化投资分析报告",
    backstory=(
        "你是一名经验丰富的投资顾问，善于整合多来源信息，在充分权衡风险与收益后"
        "给出客观、有据可查的投资建议，报告风格清晰、逻辑严谨。"
    ),
    tools=[],
    llm=llm,
    verbose=True,
)


# ── 构建 Crew ─────────────────────────────────────────────────
def build_crew(topic: str) -> Crew:
    task_research = Task(
        description=(
            f"针对「{topic}」，通过网络搜索收集最新市场动态、近期新闻、"
            "实时价格走势及宏观政策相关信息，整理成简明摘要供后续分析使用。"
        ),
        expected_output="200 字以内的市场动态摘要，包含关键数据点和信息来源。",
        agent=researcher,
    )

    task_rag = Task(
        description=(
            f"针对「{topic}」，检索本地券商研报知识库，提取机构对该标的的"
            "核心投资逻辑、财务数据、目标价及风险提示，给出专业研究结论。"
        ),
        expected_output="300 字以内的研报观点摘要，包含关键财务指标与投资评级。",
        agent=rag_analyst,
    )

    task_advise = Task(
        description=(
            f"基于市场研究员提供的实时动态和研报分析师的专业观点，"
            f"针对「{topic}」撰写一份综合投资分析报告。"
            "报告须包含：市场现状、基本面分析、风险提示、综合建议四部分。"
        ),
        expected_output="结构清晰的投资分析报告，分节呈现，结尾给出明确的综合建议。",
        agent=advisor,
        context=[task_research, task_rag],
    )

    return Crew(
        agents=[researcher, rag_analyst, advisor],
        tasks=[task_research, task_rag, task_advise],
        process=Process.sequential,
        verbose=True,
    )


# ── CLI 入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("CrewAI 多 Agent 金融投资分析系统")
    print("输入 exit 退出")
    print("=" * 60)

    while True:
        try:
            topic = input("\n请输入分析标的（如：宁德时代、新能源行业）：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出。")
            break
        if not topic:
            continue
        if topic.lower() in ("exit", "quit", "退出"):
            print("已退出。")
            break

        print(f"\n正在启动多 Agent 协作分析「{topic}」...\n")
        crew = build_crew(topic)
        result = crew.kickoff()
        print("\n" + "=" * 60)
        print("【最终投资分析报告】")
        print("=" * 60)
        print(result)
