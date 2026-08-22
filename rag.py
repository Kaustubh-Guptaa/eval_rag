import json

from langchain_chroma import Chroma
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config

embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

db = Chroma(
    collection_name=config.COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=config.CHROMA_DIR,
)


def detect_ticker(question):
    """If the question clearly names one company, return its ticker so
    retrieval can be scoped to just that filing. 
    Returns None when no company is named, or >1 is mentioned 
    (cross-company questions still need to search everything)."""
    ques = question.lower()
    
    matches = {
        ticker for ticker, aliases in config.TICKER_ALIASES.items() 
        if any(alias in ques for alias in aliases)
    } # <- Set comprehension to find matching tickers
   
    return matches.pop() if len(matches) == 1 else None

# Define LLM's answering boundary
template = """You are answering questions about SEC 10-K filings.
Use only the context below. If the answer isn't in the context, say you
don't know instead of guessing.

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model=config.CHAT_MODEL)


def format_docs(docs):
    """Join the retrieved chunks, each with a header saying where it's from."""
    parts = []
    for doc in docs:
        m = doc.metadata
        parts.append(f"[{m.get('ticker')} 10-K, FY{m.get('fiscal_year')}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# used by ask() below, which does its own retrieval via retrieve()
answer_chain = prompt | llm | StrOutputParser()


# temperature=0 so the same question keeps picking the same passages
rerank_llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)

rerank_template = """You are selecting source passages from an SEC 10-K to answer a question.

Question: {question}

Passages: {passages}

Return the passage numbers that genuinely help answer the question, ordered
most useful first. Put passages that directly state the answer before ones
that are merely on the same topic. Include at most {n}. Omit passages that
do not contribute. Reply with ONLY a JSON array of numbers, e.g. [3,1,7]."""

rerank_prompt = PromptTemplate.from_template(rerank_template)
rerank_chain = rerank_prompt | rerank_llm | StrOutputParser()


def rerank(question, candidates):
    """
    Order candidates by how useful each is for answering the question.
    
    """
    listing = "\n\n".join(f"[{i}] {doc.page_content[:600]}" for i, doc in enumerate(candidates))
    
    raw = rerank_chain.invoke(
        {
            "question": question, 
            "passages": listing, 
            "n": config.RETRIEVE_K
        }
    )

    # the model is asked for bare JSON, but tolerate it wrapping the array in
    # prose or a code fence
    try:
        start, end = raw.index("["), raw.rindex("]") + 1
        order = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return candidates[: config.RETRIEVE_K]  # fall back to embedding order

    picked, seen = [], set()
    for i in order:
        if isinstance(i, int) and 0 <= i < len(candidates) and i not in seen:
            seen.add(i) # avoid duplicates in case the LLM repeats a number
            picked.append(candidates[i])

    # If the model removed most candidates, add some back using the semantic ranking
    # so even a short answer has enough context.
    for i, doc in enumerate(candidates):
        if len(picked) >= config.RETRIEVE_K:
            break
        if i not in seen:
            picked.append(doc)

    return picked[: config.RETRIEVE_K]


def retrieve(question):
    """Fetch the chunks to answer from, scoped to one company where possible."""
    ticker = detect_ticker(question)
    
    # scoped to one company's filing - skips the other 7 entirely
    search_kwargs = {"filter": {"ticker": ticker}} if ticker else {}
    
    candidates = db.similarity_search(
        question, k=config.RERANK_FETCH_K, **search_kwargs
    )
    
    if not candidates:
        return []
    
    return rerank(question, candidates)


def ask(question):
    """Answer a question AND return the chunks used, so the app can show sources."""
    
    docs = retrieve(question)
    answer = answer_chain.invoke({"context": format_docs(docs), "question": question})
    
    return answer, docs


# Command line test: python rag.py
if __name__ == "__main__":
    question = "What are Apple's main products?"
    print(ask(question)[0])
