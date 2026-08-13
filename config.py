import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env.secrets file

# SEC requires a real name + email in the User-Agent header, otherwise
# every request comes back as a 403
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# The selected companies being indexed
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM"]

# Data Storage
RAW_DIR = "data/raw"
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "filings"

# Embedding and Chat Models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
