from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

_search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """
    搜索互联网上的实时信息。
    当需要查询新闻、时事、不在知识库中的内容时使用。
    """
    return _search.run(query)