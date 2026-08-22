import os
from dotenv import load_dotenv

# create a secrets file with your own values
# reads the .env.secrets file
load_dotenv(".env.secrets")

# SEC requires a real name + email in the User-Agent header, otherwise
# every request comes back as a 403
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# The selected companies being indexed
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM"]

# common ways someone might name each company in a question, used to scope
# retrieval to that one filing instead of searching across all 8 pooled
# together (prevents e.g. a Microsoft question pulling back a JPMorgan chunk)
TICKER_ALIASES = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "NVDA": ["nvidia", "nvda"],
    "AMZN": ["amazon", "amzn"],
    "GOOGL": ["google", "alphabet", "googl"],
    "META": ["meta", "facebook"],
    "TSLA": ["tesla", "tsla"],
    "JPM": ["jpmorgan", "jp morgan", "chase", "jpm"],
}

# Data Storage
RAW_DIR = "data/raw"
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "filings"

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini-2024-07-18"

# Retrieval: fetch a wide candidate set by embedding similarity, then have the
# LLM order it by how useful each passage actually is for the question.

# Embedding similarity finds the right neighborhood but orders it poorly.
# Widening k alone fixes recall (more of the answer ends up somewhere in the context) but
# not precision, which only rewards useful passages ranking EARLY.

RERANK_FETCH_K = 20  # candidates pulled from Chroma before reranking
RETRIEVE_K = 6       # passages actually passed to the answering model
