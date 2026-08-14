# source.py --> downloads 10-K filings from SEC EDGAR
#
# EDGAR is free to use but has two rules:
#   1. You must send a User-Agent header with your name and email,
#      otherwise you get a 403 error. Mine comes from the .env.secrets file.
#   2. Max 10 requests per second.

import time
from dataclasses import dataclass # simplifies class __init__() declarations
from pathlib import Path
import requests
import config


@dataclass
class FilingRecord:
    ticker: str
    company: str
    cik: str
    accession: str
    filing_date: str
    report_date: str
    fiscal_year: int
    primary_document: str
    path: str
    status: str


def edgar_get(url):
    """GET request with the headers SEC wants, plus a small delay."""
    time.sleep(0.2)  # keeps under the 10 requests/sec limit
    headers = {"User-Agent": config.EDGAR_USER_AGENT}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response


def resolve_tickers(tickers):
    """Turn tickers like AAPL into CIK numbers, which EDGAR uses everywhere."""
    data = edgar_get("https://www.sec.gov/files/company_tickers.json").json()

    # build a lookup table to not loop over the whole list per ticker
    lookup = {}
    for row in data.values():
        lookup[row["ticker"]] = row

    resolved = {}
    for ticker in tickers:
        if ticker not in lookup:
            print(f"Warning: ticker {ticker} not found, skipping")
            continue
        row = lookup[ticker]
        resolved[ticker] = {
            # CIKs have to be padded to 10 digits for the submissions URL
            "cik": str(row["cik_str"]).zfill(10),
            "company": row["title"],
        }
    return resolved


def latest_10k(ticker, cik, company):
    """Find the most recent 10-K for a company from its submissions feed."""
    subs = edgar_get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    recent = subs["filings"]["recent"]

    # EDGAR gives filings back as parallel lists (one list of form types,
    # one of dates, etc.), so we loop over the index
    for i in range(len(recent["form"])):
        # exact match on "10-K" also skips amendments like "10-K/A"
        if recent["form"][i] != "10-K":
            continue

        # a few filings leave reportDate blank, fall back to filingDate
        report_date = recent["reportDate"][i] or recent["filingDate"][i]

        return FilingRecord(
            ticker=ticker,
            company=company,
            cik=cik,
            accession=recent["accessionNumber"][i],
            filing_date=recent["filingDate"][i],
            report_date=report_date,
            fiscal_year=int(report_date[:4]),
            primary_document=recent["primaryDocument"][i],
            path="",
            status="pending",
        )

    print(f"No 10-K found for {ticker}")
    return None


def download_filing(record):
    """Download the 10-K HTML file and save it under data/raw/."""
    save_path = Path(config.RAW_DIR) / record.ticker / record.accession / record.primary_document

    # skip if we already downloaded this one
    if save_path.exists():
        record.path = str(save_path)
        record.status = "downloaded"
        return record

    # archive URLs want the CIK without padding and the accession without dashes
    accession_nodash = record.accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(record.cik)}/{accession_nodash}/{record.primary_document}"
    )

    response = edgar_get(url)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(response.content)

    record.path = str(save_path)
    record.status = "downloaded"
    print(f"Downloaded {record.ticker} {record.accession} ({len(response.content) // 1024} KB)")
    return record
