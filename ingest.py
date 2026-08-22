# ingest.py - the full pipeline: download filings, parse them, chunk them,
# and load everything into Chroma. Run this once before starting the app.
#
#   python ingest.py

import json
import re
from dataclasses import asdict
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

import config
import data_source

# 10-Ks are organized into numbered Items 
# (Item 1. Business, Item 1A. Risk Factors, Item 7. MD&A, ...), always in this order.
ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\.?\s*(.*)$", re.IGNORECASE)
ITEM_ORDER = [
    "1", "1A", "1B", "1C", "2", "3", "4",
    "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14", "15", "16",
]
ITEM_RANK = {key: i for i, key in enumerate(ITEM_ORDER)}


def split_into_sections(lines):
    """Break a filing's lines into (label, text) sections, one per Item.

    We do this before the character-based splitter so that chunks stay within 
    one topic. Otherwise, a fixed-size chunk could accidentally combine two 
    different sections, such as Risk Factors and Legal Proceedings.

    We use the natural ordering of Item numbers to find the real start of the 
    document, ignoring the Table of Contents and repeated page headers.
    """
    matches = []

    for i, line in enumerate(lines):
        m = ITEM_RE.match(line)

        if m:
            matches.append(
                (
                    m.group(1).upper(), i, m.group(2).strip()
                    # (Item key, line index, item title) <- Tuple
                )
            )
    
    # If the document doesn't contain anything matching ITEM_RE, return the entire filing as one section
    if not matches:
        return [("Full Filing", "\n".join(lines))]

    # Check the Ordering: The first match is Table of Contents
    # body_start is the index of the first line of the real body
    body_start = 0
    prev_rank = -1
    
    for idx, (item_key, _, _) in enumerate(matches):
        
        rank = ITEM_RANK.get(item_key)
        
        if rank is None:
            continue
        
        if rank < prev_rank:
            body_start = idx  
        
        prev_rank = rank
    # body_start stays 0 if the numbering never drops down - i.e.
    # there was no table of contents to skip past

    headers = []  # Real section headers: (item_key, line_no, title)
    current_rank = -1
    
    for item_key, line_no, title in matches[body_start:]:
        
        rank = ITEM_RANK.get(item_key)
        
        if rank is None or rank <= current_rank:
            continue  # not a recognized Item, or a repeat of the current one
        
        headers.append((item_key, line_no, title))
        current_rank = rank

    sections = []
    for idx, (item_key, line_no, title) in enumerate(headers):
        
        # Get everything from start of the Item to the end (last line) of the filing
        end = headers[idx + 1][1] if idx + 1 < len(headers) else len(lines)
        
        # Edge case: Format problem
        # Some filings have the Item title on the next line, e.g.:
        #       Item 1A.
        #       Risk Factors
        # Instead of "Item 1A. Risk Factors" on one line. If the next line is short, treat it as the title. 
        if line_no + 1 < end and (not title or len(title) <= 3):
            title = (title + lines[line_no + 1]).strip()
        
        label = f"Item {item_key}. {title}".strip()
        sections.append((label, "\n".join(lines[line_no:end])))

    return sections


# 10-Ks are filed as HTML.
# BeautifulSoup converts raw 10-K HTML into plain text
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
        separators = ["\n\n", "\n", ". ", ".", " ", ""], 
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
        
        html = Path(record.path).read_text(encoding="utf-8", errors="ignore") # ignore odd characters
        text = filing_to_text(html)
        sections = split_into_sections(text.splitlines())

        chunks = []
        metadatas = []
        for label, section_text in sections:
            for chunk in splitter.split_text(section_text):
                chunks.append(f"[{label}]\n{chunk}")
                metadatas.append(
                    {
                        "ticker": record.ticker,
                        "company": record.company,
                        "fiscal_year": record.fiscal_year,
                        "section": label,
                    }
                )

        # one filing at a time keeps each embedding request a reasonable size
        db.add_texts(chunks, metadatas=metadatas)
        record.status = "indexed"
        total += len(chunks)
        print(f"{record.ticker}: {len(chunks)} chunks indexed across {len(sections)} sections")

    # update the manifest now that everything is indexed
    with open("data/manifest.json", "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    print(f"Done - {total} chunks in {config.CHROMA_DIR}")


