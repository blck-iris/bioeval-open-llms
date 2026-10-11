"""End-to-end test of analyze.py on synthetic results (no GPU / HF access needed)."""
import json
import subprocess
import sys

import numpy as np


def _fake(model, template, acc, rng, n=120, shifts=4, bias_slot=None):
    rows = []
    for i in range(n):
        ans = int(rng.integers(4))
        for s in range(shifts):
            correct = int(rng.random() < acc)
            pred = ans if correct else int((ans + 1 + rng.integers(3)) % 4)
            ans_pos = [(j + s) % 4 for j in range(4)].index(ans)
            pred_pos = bias_slot if (bias_slot is not None and not correct) else [(j + s) % 4 for j in range(4)].index(pred)
            conf = float(np.clip(rng.normal(0.5 if correct else 0.4, 0.15), 0.26, 0.99))
            rows.append({"id": f"wmdp-bio-{i}", "benchmark": "wmdp-bio", "subtask": "bio", "n_choices": 4,
                         "answer": ans, "pred": pred, "correct": correct, "pred_pos": pred_pos,
                         "answer_pos": ans_pos, "conf": conf, "probs": [0.25] * 4,
                         "prompt_tokens": int(rng.integers(40, 300)), "template": template,
                         "shift": s, "model": model})
    return rows


def test_analyze_runs_and_writes_report(tmp_path):
    rng = np.random.default_rng(1)
    raw = tmp_path / "raw"
    raw.mkdir()
    for model, acc in [("org/small", 0.3), ("org/big", 0.6)]:
        for template in ("plain", "chat"):
            with open(raw / f"{model.replace('/', '__')}__{template}.jsonl", "w") as f:
                for r in _fake(model, template, acc, rng, bias_slot=0 if model == "org/small" else None):
                    f.write(json.dumps(r) + "\n")
    out = tmp_path / "results"
    res = subprocess.run([sys.executable, "-m", "bioeval.analyze", "--raw-dir", str(raw), "--out-dir", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    report = (out / "REPORT.md").read_text()
    assert "Accuracy" in report and "org/big" in report
    assert (out / "tables" / "accuracy.csv").exists()
    assert (out / "figures" / "accuracy_by_task.png").exists()
    import pandas as pd
    acc = pd.read_csv(out / "tables" / "accuracy.csv")
    big = acc[(acc.model == "org/big") & (acc.template == "plain")].accuracy.iloc[0]
    small = acc[(acc.model == "org/small") & (acc.template == "plain")].accuracy.iloc[0]
    assert big > small
