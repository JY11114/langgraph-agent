# 基于 LangGraph 的任务型 AI Agent

面向金融投研场景的多框架 AI Agent 系统，覆盖 LangGraph 状态图执行闭环、LangChain Tool-Calling Agent、CrewAI 多 Agent 分工协作、MCP 协议服务接入与双层记忆机制，研报 RAG 检索封装为可被多种 Agent 框架复用的独立工具。

**技术栈**：Python · LangGraph · LangChain · CrewAI · ChromaDB · FastMCP · Claude API

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                  Tool-Calling Agent（agent.py）              │
│                                                             │
│  用户提问 ─→ 长期记忆检索注入 System Prompt                   │
│  LLM 意图识别 → 工具路由（自动选择以下工具）                   │
│    ├── search_knowledge_base  本地研报 RAG 检索               │
│    ├── web_search             实时互联网搜索                  │
│    ├── get_stock_price        A 股实时行情                    │
│    ├── calculate_financial_metrics  PE/市值/增长率计算        │
│    ├── save_to_memory         写入长期记忆                    │
│    └── recall_from_memory     检索历史记忆                    │
│  ← RunnableWithMessageHistory 短期上下文管理                 │
│  ← ChromaDB 长期记忆持久化（会话结束自动摘要存储）             │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│     LangGraph 状态图（langgraph_basic.py）│
│                                          │
│  search_node（RAG 检索）                  │
│      │                                   │
│   should_retry（条件边）                  │
│      ├── 结果不足 → 重新检索（最多3次）    │
│      └── 充分 → analysis_node（LLM 分析）│
│                    │                     │
│                   END                   │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│       CrewAI 多 Agent（multi_agent.py）   │
│                                          │
│  researcher  → 网络搜索实时信息            │
│  rag_analyst → 研报知识库专业观点          │
│  advisor     → 综合两路，输出投资报告      │
│  （顺序流程，context 共享）               │
└──────────────────────────────────────────┘

mcp_server.py → FastMCP 标准协议（Claude Desktop 可直接接入）
```

## 核心功能

### 任务规划与执行闭环
基于 LangGraph 设计状态图驱动的执行流程，`StateGraph` 定义 `search → should_retry → analysis` 有向路径，条件边负责结果反思与重试（最多 3 次），实现任务分解、执行、反思、重试的闭环控制，完整状态可追踪可扩展。

### 多 Agent 分工协作
CrewAI 多角色架构：研究员（网络搜索）、研报分析师（RAG 检索）、投资顾问（综合输出），三者顺序协作并共享 context，主控 Agent 负责最终分析综合，避免单一 Agent 信息盲区。

### 自定义工具体系
将研报检索、网络搜索、A 股行情、金融计算（EPS×PE 目标价、增长率、市值）封装为 LangChain `@tool`，Tool-Calling Agent 通过 LLM 意图识别自动选择并路由，工具可被多种 Agent 框架（LangGraph / CrewAI / MCP）复用调用。

### MCP 协议接入
基于 FastMCP 实现标准 MCP 服务端，将 `search_knowledge_base`、`get_stock_price`、`web_search` 暴露为 MCP 工具，Claude Desktop 等主流 AI 工具无需额外开发即可直接接入本地知识库。

### 记忆系统
双层记忆设计：
- **短期记忆**：`RunnableWithMessageHistory` 维护单次会话完整上下文，支持跨轮引用上文信息
- **长期记忆**：ChromaDB 持久化存储，会话结束后自动摘要并写入；下次对话启动时语义检索相关历史，注入 System Prompt，跨会话延续关键结论

### RAG 工具集成
混合检索（ChromaDB 向量 + BM25 + RRF 融合）+ CrossEncoder Reranker 精排，封装为独立 LangChain Tool，可被 Tool-Calling Agent、CrewAI 各角色、LangGraph 节点统一调用。

## 项目结构

```
langgraph-agent/
├── agent.py               # Tool-Calling Agent（6 工具 · 短期+长期记忆）
├── langgraph_basic.py     # LangGraph 状态图（搜索→条件重试→分析）
├── multi_agent.py         # CrewAI 多 Agent（研究员/分析师/顾问）
├── mcp_server.py          # MCP Server（FastMCP · Claude Desktop 接入）
├── tools/
│   ├── rag_tool.py        # @tool：本地研报混合检索
│   ├── search_tool.py     # @tool：DuckDuckGo 实时搜索
│   ├── stock_tool.py      # @tool：A 股行情（akshare + 静态兜底）
│   ├── calculator_tool.py # @tool：金融指标计算（目标价/PE/增长率/市值）
│   ├── memory_tool.py     # @tool：长期记忆存取
│   └── __init__.py
├── memory/
│   └── long_term_memory.py  # ChromaDB 长期记忆（save/recall/summarize_and_save）
├── core/
│   ├── config.py          # 配置（模型名、RAG 参数、API Key）
│   ├── llm_client.py      # Claude API 封装
│   └── rag_engine.py      # RAG 引擎（向量+BM25+RRF+Reranker）
├── service/
│   └── rag_service.py     # 业务层（改写→检索→精排→生成）
├── knowledge_base/        # 4 份券商研报
│   ├── catl_research_2024.txt      # 宁德时代（华泰证券·2024-10-15）
│   ├── maotai_research_2024.txt    # 贵州茅台（中信证券·2024-11-08）
│   ├── byd_research_2024.txt       # 比亚迪（国泰君安·2024-11-12）
│   └── new_energy_sector_2024.txt  # 新能源行业策略（申万宏源·2024-11-25）
└── requirements.txt
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key
export ANTHROPIC_API_KEY=your_key_here

# Tool-Calling Agent（多工具 + 记忆系统演示）
python agent.py

# LangGraph 状态图（执行闭环 + 重试演示）
python langgraph_basic.py

# CrewAI 多 Agent 协作
python multi_agent.py

# MCP Server（供 Claude Desktop 等工具接入）
python mcp_server.py
```

## 工具体系

| 工具 | 功能 | 数据来源 |
|------|------|------|
| `search_knowledge_base` | 研报混合检索（向量+BM25+Reranker） | 本地 knowledge_base/ |
| `web_search` | 实时互联网搜索 | DuckDuckGo |
| `get_stock_price` | A 股实时行情 | akshare（兜底静态数据） |
| `calculate_financial_metrics` | 目标价/PE/增长率/市值计算 | 用户提供数据 |
| `save_to_memory` | 持久化保存关键结论 | ChromaDB |
| `recall_from_memory` | 语义检索历史记忆 | ChromaDB |

## 示例对话

```
用户：宁德时代2024年Q3毛利率是多少？储能业务增速如何？
助手：[调用 search_knowledge_base]
     根据华泰证券研报（2024-10-15）：
     • Q3 毛利率 26.3%，创近八季度新高
     • 储能电池出货 62GWh，同比增长 42%

用户：基于研报数据，帮我算一下宁德时代按 EPS=12.8、PE=16.5 的目标价
助手：[调用 calculate_financial_metrics]
     目标价 = 12.8 × 16.5 = 211.2 元

用户：把刚才宁德时代的估值结论保存到记忆里
助手：[调用 save_to_memory]
     已保存：宁德时代目标价 211.2 元（EPS=12.8，PE=16.5）
```

## 记忆系统设计

```python
# 短期记忆：RunnableWithMessageHistory 维护会话上下文
agent_with_history = RunnableWithMessageHistory(executor, get_session_history, ...)

# 长期记忆：ChromaDB 持久化
class LongTermMemory:
    def save(self, content: str, session_id: str, category: str) -> str
    def recall(self, query: str, top_k: int = 3) -> list[str]      # 语义相似检索
    def summarize_and_save(self, session_id, history, llm) -> str   # 会话结束自动摘要

# 对话启动时：检索长期记忆注入 System Prompt
memories = _long_term_memory.recall(question, top_k=3)
# 对话结束时：自动摘要写入（≥4 条消息触发）
_long_term_memory.summarize_and_save(session_id, history_dicts, llm)
```

## MCP 接入

```python
# mcp_server.py — 标准 MCP 工具暴露
mcp = FastMCP("financial-agent")

@mcp.tool()
def search_knowledge_base(query: str) -> str: ...

@mcp.tool()
def get_stock_price(stock_code: str) -> str: ...

@mcp.tool()
def web_search(query: str) -> str: ...
```

在 Claude Desktop 的 MCP 配置中添加本服务后，即可直接调用以上工具，无需额外开发。
# langgraph-agent
# langgraph-agent
