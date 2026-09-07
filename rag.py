"""Grounded generation over the knowledge base (Part 1, Task 4).

Under MOCK_LLM there is no model to judge whether an answer is grounded, so the
answer is *constructed* from retrieved text rather than written about it. Every
sentence in an answer is a sentence that came back from the index, which makes
groundedness a property of the code instead of a hope about a model.

The "I don't know" threshold is not a preset. Run `python rag.py --calibrate` to
reproduce the measurement that chose it: top-1 cosine similarity for eight
in-scope queries against three out-of-scope ones, with the threshold set between
the two clusters that were actually observed.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from chunking import split_sentences
from index_kb import embed, get_collection
from queries import IN_SCOPE, OUT_OF_SCOPE

# Measured on this corpus with all-MiniLM-L6-v2, on the 'fixed' collection that
# Task 5 recommends deploying. In-scope queries land at 0.409-0.632; the two
# clearly out-of-scope ones at 0.285 and 0.297. 0.40 sits below every real
# question and above both, and no higher value improves on it.
#
# The clusters do NOT separate cleanly. "What are your fixed deposit interest
# rates?" scores 0.466, above three genuine questions, so no similarity
# threshold refuses it without also refusing real ones. That is a property of
# the signal, not a tuning failure, and it is why Part 2 adds an output-side
# groundedness check rather than trusting this number alone.
#
# The 0.5/0.6/0.7 presets the brief warns about would refuse 4, 5 and 8 of the
# 8 real questions respectively.
SIMILARITY_THRESHOLD = 0.40

DEFAULT_STRATEGY = "fixed"
TOP_K = 3

IDK = ("I don't have anything in the knowledge base that answers this. "
       "Please route it to a human agent.")


@dataclass(frozen=True)
class Hit:
    doc_id: str
    chunk_id: str
    text: str
    similarity: float


@dataclass(frozen=True)
class Answer:
    query: str
    text: str
    grounded: bool
    top_similarity: float
    sources: tuple[str, ...]
    hits: tuple[Hit, ...]


def using_mock_llm() -> bool:
    """The brief requires every criterion to hold under MOCK_LLM alone.

    A real model can be wired in behind this flag later; nothing in Part 1
    depends on one.
    """
    return os.environ.get("MOCK_LLM", "1") != "0"


def retrieve(query: str, strategy: str = DEFAULT_STRATEGY, k: int = TOP_K) -> list[Hit]:
    res = get_collection(strategy).query(query_embeddings=embed([query]), n_results=k)
    return [
        Hit(doc_id=meta["doc_id"], chunk_id=cid, text=text, similarity=1.0 - dist)
        for cid, meta, text, dist in zip(
            res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]
        )
    ]


def _compose(query: str, hits: list[Hit]) -> str:
    """Build an answer using ONLY retrieved text.

    Deterministic and extractive: the retrieved chunks are split into sentences,
    each sentence is scored against the query, and the best ones are returned in
    the order their parent chunks were ranked. Nothing is paraphrased, so nothing
    can be invented.
    """
    query_vec = embed([query])[0]
    candidates: list[tuple[float, int, str, str]] = []
    for rank, hit in enumerate(hits):
        for sentence in split_sentences(hit.text):
            if not _is_whole_sentence(sentence):
                continue
            score = sum(a * b for a, b in zip(query_vec, embed([sentence])[0]))
            candidates.append((score, rank, sentence, hit.doc_id))

    if not candidates:
        # Every retrieved chunk was a fragment. Rather than quote half a
        # sentence, say so: a support answer that trails off mid-clause is
        # worse than an honest miss.
        return ("The knowledge base has related material but no complete "
                "statement that answers this. Please route it to a human agent.")

    best = sorted(candidates, key=lambda c: -c[0])[:3]
    best = sorted(best, key=lambda c: (c[1], -c[0]))  # restore retrieval order
    seen, parts = set(), []
    for _, _, sentence, doc_id in best:
        if sentence in seen:
            continue
        seen.add(sentence)
        parts.append(f"{sentence} [{doc_id}]")
    return " ".join(parts)


def _is_whole_sentence(text: str) -> bool:
    """Reject the half-sentences a fixed-size cut leaves behind.

    This is the cost of fixed-size chunking, and it has to be paid somewhere.
    A chunk boundary at character 200 routinely lands mid-clause, so a chunk's
    first and last "sentences" are often fragments. Quoting one into an answer
    produces text that trails off, which reads as a broken system.
    """
    text = text.strip()
    return (
        len(text) >= 40
        and text[0].isupper()
        and text.endswith((".", "!", "?"))
    )


def answer(query: str, strategy: str = DEFAULT_STRATEGY, k: int = TOP_K) -> Answer:
    hits = retrieve(query, strategy, k)
    top = hits[0].similarity if hits else 0.0

    if not hits or top < SIMILARITY_THRESHOLD:
        return Answer(query, IDK, False, top, (), tuple(hits))

    return Answer(
        query=query,
        text=_compose(query, hits),
        grounded=True,
        top_similarity=top,
        sources=tuple(dict.fromkeys(h.doc_id for h in hits)),
        hits=tuple(hits),
    )


# --------------------------------------------------------------------------
# Threshold calibration: the measurement, not a preset.
# --------------------------------------------------------------------------

def calibrate(strategy: str = DEFAULT_STRATEGY) -> None:
    print(f"Calibrating the 'I don't know' threshold on the {strategy!r} collection.")
    print("Top-1 cosine similarity per query, measured, not assumed.\n")

    in_scores, far_scores, adj_scores = [], [], []

    print("  IN-SCOPE")
    for q in IN_SCOPE:
        s = retrieve(q.text, strategy, 1)[0].similarity
        in_scores.append(s)
        print(f"    {q.query_id}  {s:.3f}   {q.text[:62]}")

    print("\n  OUT-OF-SCOPE, trivially far")
    for q in (q for q in OUT_OF_SCOPE if q.cluster == "far"):
        s = retrieve(q.text, strategy, 1)[0].similarity
        far_scores.append(s)
        print(f"    {q.query_id}  {s:.3f}   {q.text[:62]}")

    print("\n  OUT-OF-SCOPE, banking-adjacent (the real test)")
    for q in (q for q in OUT_OF_SCOPE if q.cluster == "adjacent"):
        s = retrieve(q.text, strategy, 1)[0].similarity
        adj_scores.append(s)
        print(f"    {q.query_id}  {s:.3f}   {q.text[:62]}")

    out_scores = far_scores + adj_scores
    lo_in, hi_out = min(in_scores), max(out_scores)
    print("\n  Clusters")
    print(f"    in-scope        {min(in_scores):.3f} to {max(in_scores):.3f}")
    print(f"    far             {min(far_scores):.3f} to {max(far_scores):.3f}")
    print(f"    adjacent        {min(adj_scores):.3f} to {max(adj_scores):.3f}")
    print(f"    gap             {hi_out:.3f} (highest out) to {lo_in:.3f} (lowest in)")

    if hi_out < lo_in:
        midpoint = (hi_out + lo_in) / 2
        print(f"\n    The clusters separate. Midpoint: {midpoint:.3f}")
        print(f"    Configured threshold: {SIMILARITY_THRESHOLD}")
    else:
        print(f"\n    The clusters OVERLAP. No threshold separates them cleanly;")
        print(f"    {SIMILARITY_THRESHOLD} trades one error for the other.")

    fp = [s for s in out_scores if s >= SIMILARITY_THRESHOLD]
    fn = [s for s in in_scores if s < SIMILARITY_THRESHOLD]
    print(f"\n    At {SIMILARITY_THRESHOLD}: "
          f"{len(fn)}/{len(in_scores)} in-scope wrongly refused, "
          f"{len(fp)}/{len(out_scores)} out-of-scope wrongly answered.")
    print("\n  For contrast, the tutorial presets the brief warns about:")
    for preset in (0.5, 0.6, 0.7):
        refused = sum(1 for s in in_scores if s < preset)
        print(f"    {preset}: would refuse {refused}/{len(in_scores)} real questions")


def demonstrate(strategy: str = DEFAULT_STRATEGY) -> None:
    print(f"Grounded generation, MOCK_LLM={'on' if using_mock_llm() else 'off'}, "
          f"collection {strategy!r}, top-{TOP_K}.\n")
    for q in IN_SCOPE[:5]:
        a = answer(q.text, strategy)
        print(f"{q.query_id}  {q.text}")
        print(f"    top-1 cos {a.top_similarity:.3f}  "
              f"{'GROUNDED' if a.grounded else 'FALLBACK'}  sources {list(a.sources)}")
        print(f"    {a.text}\n")

    print("Out-of-scope, which must trigger the fallback:")
    for q in OUT_OF_SCOPE:
        a = answer(q.text, strategy)
        print(f"{q.query_id}  {q.text}")
        print(f"    top-1 cos {a.top_similarity:.3f}  "
              f"{'GROUNDED' if a.grounded else 'FALLBACK'}")
        print(f"    {a.text}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibrate", action="store_true",
                   help="reproduce the threshold measurement")
    p.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=["fixed", "sentence"])
    args = p.parse_args()
    calibrate(args.strategy) if args.calibrate else demonstrate(args.strategy)


if __name__ == "__main__":
    main()
