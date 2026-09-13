"""Part 3 Task 13. The fifteen queries the RAG triad is scored on.

Selected by item_id out of eval.queries.GOLDEN_DATASET rather than copied as
strings, per D-73, so the evaluation set and the golden dataset cannot drift
apart. A typo here raises at import rather than silently scoring fourteen.

The twelve answerable items between them carry gold documents kb-01 to kb-12,
which is every required topic in the brief. The three that remain are the
negative class, and the third of them is deliberate: IU-02 is the probe of
spec 18.2 item 5 that reads 0.4645 top-1 and answers confidently from the
wrong document. A triad that cannot mark IU-02 down is not measuring
groundedness, so it is in the set rather than avoided by it.
"""

from eval.queries import GOLDEN_DATASET, GoldenItem

TRIAD_ITEM_IDS: tuple[str, ...] = (
    "EQ-01", "EQ-02", "EQ-03", "EQ-04", "EQ-05", "EQ-06",
    "EQ-07", "EQ-08", "EQ-09", "EQ-10", "EQ-11", "EQ-12",
    "FO-01",
    "FO-02",
    "IU-02",
)


def triad_items() -> list[GoldenItem]:
    """The fifteen items, in TRIAD_ITEM_IDS order."""
    by_id = {item.item_id: item for item in GOLDEN_DATASET}
    missing = [item_id for item_id in TRIAD_ITEM_IDS if item_id not in by_id]
    if missing:
        raise KeyError(
            f"TRIAD_ITEM_IDS names {missing}, which is not in GOLDEN_DATASET. "
            f"Either the id is a typo or the golden dataset dropped an item; "
            f"D-73 makes this a selection so the drift is caught here."
        )
    return [by_id[item_id] for item_id in TRIAD_ITEM_IDS]
