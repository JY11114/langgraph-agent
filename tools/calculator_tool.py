import re
from langchain.tools import tool


@tool
def calculate_financial_metrics(query: str) -> str:
    """
    计算金融估值指标，支持以下场景：
    - 目标价：给定 EPS 和 PE 倍数，计算目标价（EPS × PE）
    - 增长率：给定两期数值，计算同比增长率
    - 当前市盈率：给定股价和 EPS，反推 PE
    - 市值：给定股价和总股本，估算市值
    示例："EPS=10.2, PE=16.5" / "从441增长到510" / "股价=178, EPS=10.2"
    """
    results = []

    eps_m = re.search(r'EPS[=：\s]+(\d+\.?\d*)', query, re.IGNORECASE)
    pe_m  = re.search(r'PE[=：\s]+(\d+\.?\d*)', query, re.IGNORECASE)
    price_m  = re.search(r'股价[=：\s]+(\d+\.?\d*)', query)
    shares_m = re.search(r'(?:总股本|股本)[=：\s]+(\d+\.?\d*)\s*亿', query)

    # 目标价 = EPS × PE
    if eps_m and pe_m:
        eps = float(eps_m.group(1))
        pe  = float(pe_m.group(1))
        results.append(f"目标价 = EPS {eps} × PE {pe}x = {eps * pe:.2f} 元")

    # 当前 PE = 股价 ÷ EPS
    if price_m and eps_m and not pe_m:
        price = float(price_m.group(1))
        eps   = float(eps_m.group(1))
        results.append(f"当前 PE = 股价 {price} ÷ EPS {eps} = {price / eps:.1f}x")

    # 增长率
    g = re.search(r'(\d+\.?\d*)\s*(?:增长到|→|->|to)\s*(\d+\.?\d*)', query, re.IGNORECASE)
    if not g:
        g = re.search(r'从\s*(\d+\.?\d*)\s*(?:到|增长到)\s*(\d+\.?\d*)', query)
    if g:
        old, new = float(g.group(1)), float(g.group(2))
        if old:
            results.append(f"增长率 = ({new} - {old}) ÷ {old} × 100% = {(new - old) / old * 100:+.1f}%")

    # 市值 = 股价 × 总股本
    if price_m and shares_m:
        price  = float(price_m.group(1))
        shares = float(shares_m.group(1))
        results.append(f"市值 = {price} × {shares} 亿股 = {price * shares:.1f} 亿元")

    if not results:
        return (
            "请按以下格式输入：\n"
            "  · 目标价：EPS=10.2, PE=16.5\n"
            "  · 增长率：从441增长到510\n"
            "  · 当前PE：股价=178, EPS=10.2\n"
            "  · 市值：股价=178, 总股本=23亿"
        )

    return "\n".join(results)
