# EvalRAG
A retrieval-augmented QA system over SEC 10-K filings, designed to improve through measurement rather than guesswork. It answers questions about a company’s filing and returns the source chunks used to generate each answer. The key is the **measure → diagnose → fix → verify** loop, which improved context precision from **0.37 to 0.85**.

The system runs RAG over the latest 10-K filings from 8 companies, with ~4,900 chunks stored in Chroma. Every pipeline change is evaluated through `eval.py` using four RAGAS metrics, making retrieval improvements measurable and reproducible.

The point of the project is the evaluation discipline: two metrics grade the retriever (`context_precision`, `context_recall`), two grade the generator (`faithfulness`, `answer_relevancy`). Keeping these metrics separate helped identify *which* subsystem was broken enabling targeted fix.

## Results

Scored with RAGAS on an 8-question evaluation set.

| Iteration | Change | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|
| 0 | Baseline: naive chunking, k=4 | 0.914 | 0.773 | 0.371 | 0.739 |
| 1 | Company metadata filter + section-aware chunking | 0.850 | 0.813 | 0.434 | 0.375 |
| 2 | Widened retrieval k=4 → 8 | 0.917 | 0.934 | 0.419 | 0.844 |
| 3 | LLM reranking (20 → 6) | 0.990 | 0.935 | **0.853** | 0.781 |

Context precision went from **0.37 → 0.85** and faithfulness to **0.99**. Two honest caveats, kept here on purpose:

- Iteration 3 traded some recall (0.844 → 0.781) for higher precision.
- The eval set is only 8 questions, so every number has wide error bars. These results are directional, not definitive.

## Architecture

<img width="8192" height="4624" alt="EvalRAG-2026-08-15-194941" src="https://github.com/user-attachments/assets/3a4db784-6c03-4505-b125-d21e518b8775" />.

**Ingest (offline, run once):** SEC EDGAR → download 10-K HTML → BeautifulSoup to text → split into 10-K *Items* (sections) → `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap) → `text-embedding-3-small` → Chroma. Each chunk carries `ticker`, `company`, `fiscal_year`, and `section` metadata.

**Query (online, per question):** detect the company from the question → vector search (k=20, filtered by ticker) → LLM rerank down to the top 6 → `gpt-4o-mini` with a strict "use only this context, else say you don't know" prompt → answer + source chunks.

## Stack

Python · LangChain · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) · ChromaDB · RAGAS · FastAPI · BeautifulSoup

## Repo layout

| File | Role |
|---|---|
| `data_source.py` | EDGAR download — CIK resolution, rate limiting |
| `ingest.py` | HTML → text → sections → chunks → Chroma |
| `rag.py` | Retrieval, reranking, answer generation |
| `app.py` | FastAPI serving layer |
| `eval.py` | RAGAS scoring against the golden question set |
| `config.py` | All tunable knobs in one place |
| `data/` | Downloaded filings, `manifest.json`, Chroma DB |


