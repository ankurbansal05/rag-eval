"""Scoring. Retrieval and string metrics need no LLM; the judge does."""

import json
import re
import string

import ollama

JUDGE_MODEL = "qwen2.5:14b"


# ---------- retrieval (exact, no LLM) ----------

def is_relevant(chunk, q) -> bool:
    """A chunk is relevant if it fully contains the labelled answer span."""
    return chunk.doc_id == q.doc_id and chunk.start <= q.answer_start and chunk.end >= q.answer_end


def retrieval_scores(chunks, q) -> dict:
    ranks = [i for i, c in enumerate(chunks, 1) if is_relevant(c, q)]
    return {"recall@k": 1.0 if ranks else 0.0, "mrr": 1.0 / ranks[0] if ranks else 0.0}


# ---------- answer string match (exact, no LLM) ----------

def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_answer(answer: str, references: list[str]) -> float:
    a = _norm(answer)
    return 1.0 if any(_norm(r) in a for r in references) else 0.0


# ---------- LLM judge ----------

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "correctness": {"type": "integer", "enum": [0, 1, 2]},
        "faithfulness": {"type": "integer", "enum": [0, 1, 2]},
    },
    "required": ["reasoning", "correctness", "faithfulness"],
}

JUDGE_PROMPT = """You are a strict evaluator of a question-answering system.

Question: {question}
Reference answer(s): {references}

Retrieved context given to the system:
<<<
{context}
>>>

System answer: {answer}

Score two things:
- correctness: does the system answer mean the same as a reference answer?
  2 = yes, 1 = partially (incomplete, or right but with a wrong extra claim), 0 = no or "I don't know".
- faithfulness: is every claim in the system answer supported by the retrieved context?
  2 = fully supported, 1 = partly supported, 0 = not supported or contradicted.
  Judge this against the context only, not against the reference or your own knowledge.
  An "I don't know" answer counts as 2 (it makes no unsupported claim).

Write a brief reasoning first, then the scores."""


def judge(question: str, references: list[str], context: str, answer: str) -> dict:
    resp = ollama.generate(
        model=JUDGE_MODEL,
        prompt=JUDGE_PROMPT.format(
            question=question, references=" | ".join(references), context=context, answer=answer
        ),
        format=JUDGE_SCHEMA,
        options={"temperature": 0},
    )
    out = json.loads(resp["response"])
    return {
        "correctness": out["correctness"] / 2,
        "faithfulness": out["faithfulness"] / 2,
        "judge_reasoning": out["reasoning"],
    }
