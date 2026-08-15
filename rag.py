from langchain_chroma import Chroma
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config

embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

db = Chroma(
    collection_name=config.COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=config.CHROMA_DIR,
)

retriever = db.as_retriever(search_kwargs={"k": 4}) # k nearest neighbors to retrieve

# defining LLM's answering boundary
# explicitly instructing to say "I don't know" when the question is out of context
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


# the complete RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# same thing minus the retriever, used by ask() below
# used in the app
answer_chain = prompt | llm | StrOutputParser()


def ask(question):
    """Answer a question AND return the chunks used, so the app can show sources."""
    docs = retriever.invoke(question)
    answer = answer_chain.invoke({"context": format_docs(docs), "question": question})
    return answer, docs


# Command line test: python rag.py 
if __name__ == "__main__":
    question = "What are Apple's main products?"
    print(chain.invoke(question))
