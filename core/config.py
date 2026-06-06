import os

class Config:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    MODEL_NAME = "claude-haiku-4-5"
    MAX_TOKENS = 1024
    MAX_HISTORY_LENGTH = 20

    SYSTEM_PROMPT = "你是一个金融助手，请用简洁准确的话回答。"

    # RAG 配置（知识库目录）
    KNOWLEDGE_BASE_DIR = "knowledge_base"
    COLLECTION_NAME = "financial_kb"
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 80
    TOP_K = 5

    RAG_SYSTEM_PROMPT = """你是一位专业的金融分析师助手，根据以下研报内容回答问题。
如果研报中没有相关信息，请直接说"研报中未提及该内容"，不要编造数据。"""
