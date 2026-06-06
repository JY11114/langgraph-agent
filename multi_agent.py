"""
基于 CrewAI 的多 Agent 金融分析系统

角色分工：
  researcher  — 网络搜索，收集实时新闻与市场数据
  rag_analyst — 检索本地金融研报知识库，提取专业观点
  advisor     — 综合两路信息，输出投资分析结论
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import Field
from langchain_community.tools import DuckDuckGoSearchRun
from service.rag_service import RAGService

# ── 工具封装（CrewAI 要求继承 BaseTool）─────────────────────

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "搜索互联网实时信息，适用于最新新闻、市场动态、公司公告等。输入搜索关键词。"

    def _run(self, query: str) -> str:
        return DuckDuckGoSearchRun().run(query)


class RAGSearchTool(BaseTool):
    name: str = "rag_search"
    description: str = "检索本地金融研报知识库，适用于行业分析、券商观点、研究报告等专业内容。输入查询问题。"
    _rag: RAGService = None

    def model_post_init(self, __context):
        self._rag = RAGService()

    def _run(self, query: str) -> str:
        try:
            return self._rag.chat(query)
        except Exception as e:
            return f"知识库查询失败：{e}"


# ── Agent 定义 ────────────────────────────────────────────────

researcher = Agent(
    role="金融研究员",
    goal="通过网络搜索收集关于目标公司或行业的最新新闻、财务数据和市场动态",
    backstory="你是一位经验丰富的金融信息收集专家，擅长从互联网快速定位高价值信息。",
    tools=[WebSearchTool()],
    verbose=True,
    allow_delegation=False,
)

rag_analyst = Agent(
    role="研报分析师",
    goal="从本地金融研报知识库中检索相关专业分析，提取券商观点和行业洞察",
    backstory="你是专注于研读券商研报的分析师，熟悉从专业文献中提炼核心投资逻辑。",
    tools=[RAGSearchTool()],
    verbose=True,
    allow_delegation=False,
)

advisor = Agent(
    role="投资顾问",
    goal="综合网络信息与研报观点，给出结构化的投资分析结论",
    backstory="你是资深投资顾问，擅长整合多源信息并输出逻辑清晰、有据可查的投资建议。",
    tools=[],
    verbose=True,
    allow_delegation=False,
)


# ── 任务定义 ──────────────────────────────────────────────────

def build_crew(question: str) -> Crew:
    t1 = Task(
        description=f"针对问题「{question}」，搜索互联网，收集最新相关新闻、数据和市场动态，整理成结构化摘要。",
        expected_output="包含来源的要点列表，覆盖最新动态、关键数据和市场情绪。",
        agent=researcher,
    )

    t2 = Task(
        description=f"针对问题「{question}」，检索本地研报知识库，提取相关的专业分析、券商评级和行业观点。",
        expected_output="来自研报的专业观点摘要，包含具体数据或结论。",
        agent=rag_analyst,
    )

    t3 = Task(
        description=(
            f"基于研究员收集的实时信息和研报分析师提取的专业观点，"
            f"对问题「{question}」给出综合投资分析报告。"
            f"报告需包含：核心结论、支撑依据、潜在风险。"
        ),
        expected_output="结构化投资分析报告，含结论、依据和风险提示，字数 300 字以上。",
        agent=advisor,
        context=[t1, t2],
    )

    return Crew(
        agents=[researcher, rag_analyst, advisor],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=True,
    )


# ── 入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        "宁德时代近期值得投资吗",
        "新能源行业2026年的投资机会在哪里",
    ]

    for q in questions:
        print("\n" + "=" * 60)
        print(f"问题：{q}")
        print("=" * 60)
        crew = build_crew(q)
        result = crew.kickoff()
        print("\n最终报告：")
        print(result)
