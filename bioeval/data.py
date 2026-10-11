"""Dataset loaders. Every loader returns a list of items with a common schema:

    {"id": str, "benchmark": str, "subtask": str, "question": str,
     "choices": list[str], "answer": int}

Only public benchmarks are used:
  * WMDP-Bio    (Li et al., 2024)        -> HF: cais/wmdp, config "wmdp-bio"
  * LAB-Bench   (Laurent et al., 2024)   -> HF: futurehouse/lab-bench, text-only subtasks

NOTE: column names below follow the public dataset cards. If a dataset revision renames a
column, adjust the small mapping in the loader (all column access is in this file).
"""
from __future__ import annotations

import hashlib
import random
from typing import Optional

INSUFFICIENT = "Insufficient information to answer the question"

# Text-only LAB-Bench subtasks (the figure/table/supplement subtasks need images or files).
LABBENCH_TEXT_SUBTASKS = ["LitQA2", "SeqQA", "CloningScenarios", "ProtocolQA", "DbQA"]


def _load(path: str, name: Optional[str], split: str):
    from datasets import load_dataset  # lazy import keeps unit tests light

    return load_dataset(path, name, split=split) if name else load_dataset(path, split=split)


def load_wmdp_bio(limit: Optional[int] = None) -> list[dict]:
    ds = _load("cais/wmdp", "wmdp-bio", "test")
    items = []
    for i, r in enumerate(ds):
        items.append(
            {
                "id": f"wmdp-bio-{i}",
                "benchmark": "wmdp-bio",
                "subtask": "bio",
                "question": r["question"],
                "choices": list(r["choices"]),
                "answer": int(r["answer"]),
            }
        )
    return items[:limit] if limit else items


def _stable_rng(key: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return random.Random(int(digest[:12], 16))


def load_labbench(subtask: str, limit: Optional[int] = None, seed: int = 0,
                  add_insufficient: bool = False) -> list[dict]:
    """LAB-Bench stores `ideal` (correct) and `distractors`; we build shuffled choices.

    Shuffling is deterministic per question id so results are reproducible. If
    `add_insufficient` is set, the official 'Insufficient information' option is appended
    (after shuffling), as in the LAB-Bench paper's evaluation protocol.
    """
    ds = _load("futurehouse/lab-bench", subtask, "train")
    items = []
    for i, r in enumerate(ds):
        qid = str(r.get("id", i))
        question = r["question"]
        if r.get("protocol"):  # ProtocolQA ships the protocol text separately
            question = f"Protocol:\n{r['protocol']}\n\nQuestion: {question}"
        options = [r["ideal"]] + list(r["distractors"])
        rng = _stable_rng(qid, seed)
        order = list(range(len(options)))
        rng.shuffle(order)
        choices = [options[j] for j in order]
        answer = order.index(0)
        if add_insufficient:
            choices.append(INSUFFICIENT)
        items.append(
            {
                "id": f"labbench-{subtask}-{qid}",
                "benchmark": "lab-bench",
                "subtask": subtask,
                "question": question,
                "choices": choices,
                "answer": answer,
            }
        )
    return items[:limit] if limit else items


def load_benchmarks(spec: list[str], limit: Optional[int] = None, seed: int = 0,
                    add_insufficient: bool = False) -> list[dict]:
    """spec examples: ["wmdp-bio", "labbench:LitQA2", "labbench:SeqQA"]"""
    items: list[dict] = []
    for s in spec:
        try:
            if s == "wmdp-bio":
                items += load_wmdp_bio(limit)
            elif s.startswith("labbench:"):
                items += load_labbench(s.split(":", 1)[1], limit, seed, add_insufficient)
            else:
                raise ValueError(f"Unknown benchmark spec: {s}")
        except Exception as e:  # keep going so one broken subtask does not kill a long run
            print(f"[warn] could not load {s}: {type(e).__name__}: {e}")
    return items
