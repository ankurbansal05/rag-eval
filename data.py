"""Load a small SQuAD subset: a corpus of articles plus labelled questions.

Each article's paragraphs are joined into one document, so chunking can cross
paragraph boundaries like it would on real documents. Every question keeps the
character span of its answer inside that document, which is how we know which
chunk is the "correct" one to retrieve.
"""

import random
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class Doc:
    id: str
    title: str
    text: str


@dataclass
class Question:
    id: str
    doc_id: str
    question: str
    answers: list[str]  # all acceptable reference answers
    answer_start: int  # char offset of the first answer in the doc
    answer_end: int


def load_squad(n_articles: int = 5, n_questions: int = 50, seed: int = 42):
    ds = load_dataset("rajpurkar/squad", split="validation")

    titles = sorted(set(ds["title"]))
    random.Random(seed).shuffle(titles)
    chosen = set(titles[:n_articles])

    docs: dict[str, Doc] = {}
    para_offset: dict[tuple[str, str], int] = {}
    questions: list[Question] = []

    for row in ds:
        title = row["title"]
        if title not in chosen:
            continue
        doc = docs.setdefault(title, Doc(id=title, title=title.replace("_", " "), text=""))
        key = (title, row["context"])
        if key not in para_offset:
            if doc.text:
                doc.text += "\n\n"
            para_offset[key] = len(doc.text)
            doc.text += row["context"]

        start = para_offset[key] + row["answers"]["answer_start"][0]
        questions.append(
            Question(
                id=row["id"],
                doc_id=title,
                question=row["question"],
                answers=list(dict.fromkeys(row["answers"]["text"])),
                answer_start=start,
                answer_end=start + len(row["answers"]["text"][0]),
            )
        )

    random.Random(seed).shuffle(questions)
    return list(docs.values()), questions[:n_questions]
