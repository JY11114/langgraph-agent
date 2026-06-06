from langchain.tools import tool

# 扩展兜底价格表（当 akshare 不可用时使用）
_FALLBACK_PRICES = {
    "600519": ("贵州茅台", "1680.00", "+0.82%"),
    "000001": ("平安银行", "12.50", "-0.32%"),
    "300750": ("宁德时代", "178.30", "+1.24%"),
    "000858": ("五粮液", "138.20", "+0.58%"),
    "601318": ("中国平安", "45.60", "-0.44%"),
    "002594": ("比亚迪", "295.00", "+2.11%"),
    "600036": ("招商银行", "38.20", "+0.26%"),
    "601166": ("兴业银行", "19.85", "-0.15%"),
    "000333": ("美的集团", "58.40", "+0.69%"),
    "601899": ("紫金矿业", "15.30", "+1.32%"),
    "300274": ("阳光电源", "95.60", "+3.21%"),
    "000568": ("泸州老窖", "142.50", "+0.35%"),
    "600900": ("长江电力", "28.70", "+0.42%"),
    "601088": ("中国神华", "39.50", "-0.25%"),
    "300014": ("亿纬锂能", "42.30", "+1.87%"),
}


def _fetch_stock_price(stock_code: str) -> dict:
    """内部函数：优先用 akshare 实时行情，失败时降级到静态数据。"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == stock_code]
        if not row.empty:
            r = row.iloc[0]
            return {
                "name": r["名称"],
                "price": f"{r['最新价']:.2f}",
                "change_pct": f"{r['涨跌幅']:+.2f}%",
                "source": "实时行情（东方财富）",
            }
    except Exception:
        pass  # akshare 不可用时静默降级

    if stock_code in _FALLBACK_PRICES:
        name, price, change = _FALLBACK_PRICES[stock_code]
        return {"name": name, "price": price, "change_pct": change, "source": "静态参考价（非实时）"}

    return {"name": "未知", "price": "N/A", "change_pct": "N/A", "source": "未找到"}


@tool
def get_stock_price(stock_code: str) -> str:
    """
    查询 A 股股票当前价格。
    输入 6 位股票代码，例如 '600519'（贵州茅台）、'300750'（宁德时代）、'002594'（比亚迪）。
    优先返回实时行情，不可用时返回参考价格。
    """
    info = _fetch_stock_price(stock_code)
    if info["price"] == "N/A":
        return f"未找到股票代码 {stock_code}，请检查代码是否正确（A股为6位数字）。"
    return (
        f"{info['name']}（{stock_code}）\n"
        f"当前价：{info['price']} 元｜涨跌幅：{info['change_pct']}\n"
        f"数据来源：{info['source']}"
    )
