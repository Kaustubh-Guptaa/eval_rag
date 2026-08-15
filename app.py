# app.py - small FastAPI server that puts a web page in front of the chain
#
#   uvicorn app:app --reload
#   then open http://localhost:8000

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import ask

app = FastAPI(title="EvalRAG")


class Question(BaseModel):
    question: str


@app.post("/api/ask")
def ask_endpoint(q: Question):
    answer, docs = ask(q.question)

    # send back a trimmed version of each source chunk for the sources panel
    sources = []
    for doc in docs:
        sources.append(
            {
                "ticker": doc.metadata.get("ticker"),
                "company": doc.metadata.get("company"),
                "fiscal_year": doc.metadata.get("fiscal_year"),
                "text": doc.page_content[:300],
            }
        )

    return {"answer": answer, "sources": sources}


# serve the chat page (this has to come after the /api routes)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
