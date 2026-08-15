# ingest.py - the full pipeline: download filings, parse them, chunk them,
# and load everything into Chroma. Run this once before starting the app.
#
#   python ingest.py

import json
from dataclasses import asdict
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

import config
import data_source


# 10-Ks are filed as HTML. Therefore, we use BeautifulSoup.
# Convert raw 10-K HTML into plain text
def filing_to_text(html):
    
    soup = BeautifulSoup(html, "html.parser")

    # remove script/style HTML tags
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # the raw output has blank lines and stray whitespace,
    # clean them line by line
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    
    if not config.EDGAR_USER_AGENT:
        raise SystemExit("Set EDGAR_USER_AGENT in .env (SEC returns 403 without it)")
    if not config.OPENAI_API_KEY:
        raise SystemExit("Set OPENAI_API_KEY in .env (required for embeddings)")

    # step 1: download the latest 10-K for each ticker
    companies = data_source.resolve_tickers(config.TICKERS)

    records = []
    for ticker, info in companies.items():
        record = data_source.latest_10k(ticker, info["cik"], info["company"])
        if record:
            record = data_source.download_filing(record)
            records.append(record)

    # keep a manifest to audit what's in the index
    Path("data").mkdir(exist_ok=True)
    with open("data/manifest.json", "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    # step 2 + 3: parse, chunk, and embed each filing into Chroma (Vector DB)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], 
        chunk_size = 1000, 
        chunk_overlap = 150
    )

    openai_embedding = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    db = Chroma(
        collection_name = config.COLLECTION_NAME,
        embedding_function = openai_embedding,
        persist_directory = config.CHROMA_DIR,
    )

    total = 0
    for record in records:
        # errors="ignore" because a few filings have odd characters in them
        html = Path(record.path).read_text(encoding="utf-8", errors="ignore")
        text = filing_to_text(html)
        chunks = splitter.split_text(text)

        # every chunk remembers which company and year it came from,
        # so the app can show sources later
        metadatas = []
        for _ in chunks:
            metadatas.append(
                {
                    "ticker": record.ticker,
                    "company": record.company,
                    "fiscal_year": record.fiscal_year,
                }
            )

        # one filing at a time keeps each embedding request a reasonable size
        db.add_texts(chunks, metadatas=metadatas)
        record.status = "indexed"
        total += len(chunks)
        print(f"{record.ticker}: {len(chunks)} chunks indexed")

    # update the manifest now that everything is indexed
    with open("data/manifest.json", "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    print(f"Done - {total} chunks in {config.CHROMA_DIR}")


