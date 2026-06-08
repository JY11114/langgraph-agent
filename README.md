# langgraph-agent

金融研报分析 Agent，本地 RAG 检索 + CrewAI 联网搜索两路召回，LangGraph 状态图控制执行流程，LLM 质量打分驱动重试。另有 CrewAI 多角色协作版本和 MCP Server 供 Claude Desktop 接入。

**技术栈**：Python、LangGraph、LangChain、CrewAI、MCP、Tool Calling、ChromaDB、Claude API

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                  Tool-Calling Agent（agent.py）              │
│                                                             │
│  用户提问 → 长期记忆检索注入 System Prompt                    │
│  LLM 意图识别 → 工具路由                                     │
│    ├── search_knowledge_base  本地研报 RAG 检索               │
│    ├── web_search             实时互联网搜索                  │
│    ├── get_stock_price        A 股实时行情                    │
│    ├── calculate_financial_metrics  PE/市值/增长率计算        │
│    ├── save_to_memory         写入长期记忆                    │
│    └── recall_from_memory     检索历史记忆                    │
│  ← MemorySaver 短期上下文管理                                │
│  ← ChromaDB 长期记忆持久化（会话结束自动摘要存储）             │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│     LangGraph 状态图（langgraph_basic.py）                 │
│                                                          │
│  rag_search（向量+BM25+RRF+Reranker 混合检索）             │
│      │                                                   │
│  web_search（CrewAI Agent + DuckDuckGo）                 │
│      │                                                   │
│   should_retry（条件边）                                  │
│      ├── 召回不足 → rag_search（最多重试 3 次）             │
│      └── 充分 → analysis（LLM 综合分析）                  │
│                    │                                     │
│              quality_check（LLM 对结果打分 1-10）         │
│                    │                                     │
│               should_redo（条件边）                       │
│                    ├── 分数<6 → analysis（最多重做 2 次）  │
│                    └── 达标 → save_memory → END          │
└──────────────────────────────────────────────────────────┘

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

## 模块说明

### langgraph_basic.py
LangGraph 状态图主文件。五个节点：`rag_search`→`web_search`→`analysis`→`quality_check`→`save_memory`，两条条件边分别控制搜索重试和分析重做。RAG 用向量+BM25+RRF 混合检索后 Reranker 精排，联网搜索由 CrewAI Agent 驱动。质量达标后把本次分析结论写入长期记忆。

### agent.py
Tool-Calling Agent，直接把 6 个工具挂给 LLM，MemorySaver 管短期上下文，ChromaDB 持久化长期记忆。每次问答前把相关历史记忆拼入消息前缀，会话满 4 条后自动提炼摘要写入。

### multi_agent.py
CrewAI 版本。researcher/rag_analyst/advisor 三角色顺序执行，context 共享，最终由 advisor 汇总成投资分析报告。

### mcp_server.py
FastMCP 封装，暴露三个工具（知识库检索、股价查询、联网搜索），Claude Desktop 通过 stdio 接入。

## 项目结构

```
langgraph-agent/
├── agent.py               # Tool-Calling Agent（6 工具 · 短期+长期记忆）
├── langgraph_basic.py     # LangGraph 状态图（RAG+联网→分析→质量控制）
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
pip install -r requirements.txt

export ANTHROPIC_API_KEY=your_key_here

# Tool-Calling Agent（多工具 + 记忆系统）
python agent.py

# LangGraph 状态图（执行闭环 + 重试）
python langgraph_basic.py

# CrewAI 多 Agent 协作
python multi_agent.py

# MCP Server
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
     · Q3 毛利率 26.3%，创近八季度新高
     · 储能电池出货 62GWh，同比增长 42%

用户：基于研报数据，帮我算一下宁德时代按 EPS=12.8、PE=16.5 的目标价
助手：[调用 calculate_financial_metrics]
     目标价 = 12.8 × 16.5 = 211.2 元

用户：把刚才宁德时代的估值结论保存到记忆里
助手：[调用 save_to_memory]
     已保存：宁德时代目标价 211.2 元（EPS=12.8，PE=16.5）
```

## 记忆系统

```python
# 短期记忆：MemorySaver 维护会话上下文
_agent = create_agent(llm, tools, checkpointer=MemorySaver())

# 长期记忆：ChromaDB 持久化
class LongTermMemory:
    def save(self, content: str, session_id: str, category: str) -> str
    def recall(self, query: str, top_k: int = 3) -> list[str]
    def summarize_and_save(self, session_id, history, llm) -> str

# 启动时检索长期记忆注入提示词前缀
memories = _long_term_memory.recall(question, top_k=3)
# 会话满 4 条后自动摘要写入
_long_term_memory.summarize_and_save(session_id, history_dicts, _llm_client)
```

## MCP 接入

```python
mcp = FastMCP("financial-tools")

@mcp.tool()
def search_knowledge_base(query: str) -> str: ...

@mcp.tool()
def get_stock_price(stock_code: str) -> str: ...

@mcp.tool()
def web_search(query: str) -> str: ...
```

Claude Desktop 配置（`claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "financial-tools": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"]
    }
  }
}
```
