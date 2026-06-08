from langchain.tools import tool


def _code_to_yahoo(stock_code: str) -> str:
    """A 股代码转 Yahoo Finance 格式：60xxxx → .SS，其余 → .SZ。"""
    suffix = ".SS" if stock_code.startswith("6") else ".SZ"
    return stock_code + suffix


def _fetch_stock_price(stock_code: str) -> dict:
    try:
        import yfinance as yf
        ticker = yf.Ticker(_code_to_yahoo(stock_code))
        info = ticker.fast_info
        price = info["lastPrice"]
        prev_close = info["previousClose"]
        change_pct = (price - prev_close) / prev_close * 100
        name = ticker.info.get("longName") or ticker.info.get("shortName") or stock_code
        return {
            "name": name,
            "price": f"{price:.2f}",
            "change_pct": f"{change_pct:+.2f}%",
            "source": "实时行情（Yahoo Finance）",
        }
    except Exception:
        pass

    _FALLBACK = {
        "600519": "贵州茅台",
        "300750": "宁德时代",
        "002594": "比亚迪",
        "000858": "五粮液",
    }
    return {
        "name": _FALLBACK.get(stock_code, "未知"),
        "price": "N/A",
        "change_pct": "N/A",
        "source": "行情获取失败",
    }


@tool
def get_stock_price(stock_code: str) -> str:
    """
    查询 A 股股票当前价格。
    输入 6 位股票代码，例如 '600519'（贵州茅台）、'300750'（宁德时代）、'002594'（比亚迪）。
    """
    info = _fetch_stock_price(stock_code)
    if info["price"] == "N/A":
        return f"{info['name']}（{stock_code}）行情获取失败，请稍后重试。"
    return (
        f"{info['name']}（{stock_code}）\n"
        f"当前价：{info['price']} 元｜涨跌幅：{info['change_pct']}\n"
        f"数据来源：{info['source']}"
    )
