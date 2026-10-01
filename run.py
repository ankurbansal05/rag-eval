"""Run the RAG pipeline under several configs and compare scores.

    uv run run.py                   # full run with the LLM judge
    uv run run.py --no-judge        # retrieval + string-match only (fast)
    uv run run.py --questions 10    # quick smoke test
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from data import load_squad
from metrics import contains_answer, judge, retrieval_scores
from rag import Index, generate

# Edit this list to compare other settings.
CONFIGS = [
    {"chunk_words": 50, "overlap_words": 10, "k": 3},
    {"chunk_words": 150, "overlap_words": 30, "k": 3},
    {"chunk_words": 400, "overlap_words": 50, "k": 3},
    {"chunk_words": 150, "overlap_words": 30, "k": 5},
]

N_CANARIES = 5  # deliberately wrong answers used to test the judge

RESULTS = Path("results")


def name(cfg):
    return f"chunk{cfg['chunk_words']}_ov{cfg['overlap_words']}_k{cfg['k']}"


def run_config(cfg, docs, questions, use_judge):
    t0 = time.time()
    index = Index(docs, cfg["chunk_words"], cfg["overlap_words"])
    rows = []

    # Phase 1: retrieve + generate with the small model.
    for i, q in enumerate(questions, 1):
        chunks = index.retrieve(q.question, cfg["k"])
        answer = generate(q.question, chunks)
        rows.append({
            "id": q.id,
            "question": q.question,
            "references": q.answers,
            "answer": answer,
            "retrieved": [c.id for c in chunks],
            "context": "\n\n---\n\n".join(c.text for c in chunks),
            **retrieval_scores(chunks, q),
            "contains_answer": contains_answer(answer, q.answers),
        })
        print(f"\r  generating {i}/{len(questions)}", end="", flush=True)
    print()

    # Phase 2: judge with the bigger model (kept separate so the two
    # models aren't swapped in and out of memory on every question).
    if use_judge:
        for i, r in enumerate(rows, 1):
            r.update(judge(r["question"], r["references"], r["context"], r["answer"]))
            print(f"\r  judging {i}/{len(rows)}", end="", flush=True)
        print()

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{name(cfg)}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    metrics = ["recall@k", "mrr", "contains_answer"] + (["correctness", "faithfulness"] if use_judge else [])
    summary = {m: mean(r[m] for r in rows) for m in metrics}
    summary["chunks"] = len(index.chunks)
    summary["minutes"] = (time.time() - t0) / 60
    return rows, summary


def canary_check(rows):
    """Judge answers we know are wrong: each question gets another question's answer.
    A trustworthy judge should score these near 0 on correctness."""
    scores = []
    for i in range(min(N_CANARIES, len(rows) - 1)):
        r, other = rows[i], rows[i + 1]
        scores.append(judge(r["question"], r["references"], r["context"], other["references"][0]))
    return mean(s["correctness"] for s in scores), mean(s["faithfulness"] for s in scores)


def print_table(summaries):
    cols = list(next(iter(summaries.values())).keys())
    header = f"{'config':<24}" + "".join(f"{c:>17}" for c in cols)
    print("\n" + header + "\n" + "-" * len(header))
    for cfg_name, s in summaries.items():
        cells = "".join(f"{s[c]:>17.0f}" if c == "chunks" else f"{s[c]:>17.2f}" for c in cols)
        print(f"{cfg_name:<24}{cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=50)
    ap.add_argument("--articles", type=int, default=5)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    docs, questions = load_squad(args.articles, args.questions)
    print(f"Corpus: {len(docs)} articles, {sum(len(d.text) for d in docs):,} chars. Questions: {len(questions)}")

    summaries, first_rows = {}, None
    for cfg in CONFIGS:
        print(f"\n== {name(cfg)} ==")
        rows, summaries[name(cfg)] = run_config(cfg, docs, questions, not args.no_judge)
        first_rows = first_rows or rows

    print_table(summaries)
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    if not args.no_judge:
        c, fa = canary_check(first_rows)
        print(f"\nJudge sanity check ({N_CANARIES} deliberately wrong answers):")
        print(f"  correctness {c:.2f} (should be near 0), faithfulness {fa:.2f}")
        if c > 0.3:
            print("  WARNING: the judge is scoring wrong answers as correct; don't trust its scores.")

    print(f"\nPer-question results are in {RESULTS}/")


if __name__ == "__main__":
    main()
