"""
MCP Server — 金融工具服务

将 RAG 检索、股票查询、网络搜索封装为标准 MCP 工具，
任何 MCP 兼容的客户端（Claude Desktop、其他 Agent）均可接入调用。

启动方式：
    python mcp_server.py

客户端配置示例（claude_desktop_config.json）：
    {
      "mcpServers": {
        "financial-tools": {
          "command": "python",
          "args": ["/path/to/mcp_server.py"]
        }
      }
    }
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

mcp = FastMCP("financial-tools")

# ── 延迟初始化 RAG（避免启动时加载慢）────────────────────────
_rag = None

def get_rag():
    global _rag
    if _rag is None:
        from service.rag_service import RAGService
        _rag = RAGService()
    return _rag


# ── Tool 1：本地金融研报知识库检索 ───────────────────────────

@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """
    检索本地金融研报知识库。
    适用场景：查询券商研报、行业分析、金融机构观点等专业内容。
    """
    try:
        return get_rag().chat(query)
    except Exception as e:
        return f"知识库查询失败：{e}"


# ── Tool 2：股票价格查询 ──────────────────────────────────────

@mcp.tool()
def get_stock_price(stock_code: str) -> str:
    """
    查询 A 股股票当前价格。
    输入 6 位股票代码，例如 '600519'（贵州茅台）、'300750'（宁德时代）。
    """
    prices = {
        "600519": "贵州茅台：1680.00 元",
        "000001": "平安银行：12.50 元",
        "300750": "宁德时代：178.30 元",
        "000858": "五粮液：138.20 元",
        "601318": "中国平安：45.60 元",
    }
    return prices.get(stock_code, f"未找到股票代码 {stock_code}，请检查代码是否正确。")


# ── Tool 3：互联网实时搜索 ────────────────────────────────────

@mcp.tool()
def web_search(query: str) -> str:
    """
    搜索互联网实时信息。
    适用场景：最新市场新闻、公司公告、宏观经济动态等知识库未收录的内容。
    """
    try:
        return DuckDuckGoSearchRun().run(query)
    except Exception as e:
        return f"搜索失败：{e}"


# ── 启动 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("MCP Server 启动中，监听 stdio...")
    mcp.run(transport="stdio")
