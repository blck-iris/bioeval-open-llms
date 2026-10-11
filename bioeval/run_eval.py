"""Command-line entry point.

Example (Colab T4):
    python -m bioeval.run_eval \
        --models Qwen/Qwen2.5-1.5B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --benchmarks wmdp-bio labbench:LitQA2 labbench:SeqQA \
        --templates plain chat --n-shifts 4
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from .data import load_benchmarks
from .scoring import load_model, score_items

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]
DEFAULT_BENCHMARKS = ["wmdp-bio", "labbench:LitQA2", "labbench:SeqQA", "labbench:CloningScenarios"]


def slug(name: str) -> str:
    return name.replace("/", "__")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    ap.add_argument("--templates", nargs="+", default=["plain"], choices=["plain", "chat"])
    ap.add_argument("--n-shifts", type=int, default=1,
                    help="number of cyclic option-order rotations (4 puts the correct answer in every slot of a 4-way question)")
    ap.add_argument("--limit", type=int, default=None, help="max items per benchmark/subtask (use ~100 for a smoke test)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--load-in-4bit", action="store_true", help="4-bit quantisation (for 7B models on a T4)")
    ap.add_argument("--add-insufficient", action="store_true",
                    help="append LAB-Bench's 'Insufficient information' option")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/raw")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_benchmarks(args.benchmarks, args.limit, args.seed, args.add_insufficient)
    print(f"Loaded {len(items)} items from {args.benchmarks}")
    if not items:
        raise SystemExit("No items loaded; check dataset access and names.")

    import torch

    for model_name in args.models:
        for template in args.templates:
            out_path = out_dir / f"{slug(model_name)}__{template}.jsonl"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {out_path} exists (use --overwrite)")
                continue
            print(f"\n=== {model_name} | template={template} ===")
            t0 = time.time()
            model = tok = None
            try:
                model, tok = load_model(model_name, args.load_in_4bit)
                records = []
                for shift in range(args.n_shifts):
                    records += score_items(model, tok, items, template, shift, args.batch_size, args.max_length)
            except Exception as e:
                print(f"[error] {model_name}/{template}: {type(e).__name__}: {e}")
                records = None
            finally:
                del model, tok
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if records is None:
                continue
            meta = {"model": model_name, "template": template, "n_shifts": args.n_shifts,
                    "seed": args.seed, "seconds": round(time.time() - t0, 1)}
            with open(out_path, "w") as f:
                for r in records:
                    r["model"] = model_name
                    f.write(json.dumps(r) + "\n")
            (out_dir / f"{slug(model_name)}__{template}.meta.json").write_text(json.dumps(meta, indent=2))
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"saved {out_path}  overall acc={acc:.3f}  ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
