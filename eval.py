# eval.py - runs my golden questions through the chain and scores the
# answers with RAGAS. This is how I check whether a change (different
# chunk size, different prompt, different k) actually made things better
# instead of just eyeballing a few answers.
#
#   python eval.py

import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

import config
from rag import ask

if __name__ == "__main__":
    
    with open("data/eval_questions.json") as f:
        eval_questions = json.load(f)

    # run every question through the real chain
    rows = []
    for ques in eval_questions:
        print("Q:", ques["question"])
        answer, docs = ask(ques["question"])
        rows.append(
            {
                "user_input": ques["question"],
                "response": answer,
                "retrieved_contexts": [d.page_content for d in docs],
                "reference": ques["reference"],
            }
        )

    dataset = EvaluationDataset.from_list(rows)

    # RAGAS uses an LLM as the judge. Here, we use the same model again
    result = evaluate(
        dataset,
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall],
        llm = LangchainLLMWrapper(ChatOpenAI(model=config.CHAT_MODEL)),
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=config.EMBEDDING_MODEL)),
    )

    print("\n", result)

    # save the raw answers to audit model responses
    with open("data/eval_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("Saved answers to data/eval_results.json")
