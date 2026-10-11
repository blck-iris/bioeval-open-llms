"""Analysis of saved results: accuracy with CIs, position bias, order robustness, calibration,
template sensitivity, cross-model agreement, and difficulty/length effects.

Usage:  python -m bioeval.analyze --raw-dir results/raw --out-dir results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .metrics import cohen_kappa, expected_calibration_error, wilson_interval


def load_results(raw_dir: str) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(raw_dir).glob("*.jsonl")):
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    if not rows:
        raise SystemExit(f"No .jsonl results in {raw_dir}")
    df = pd.DataFrame(rows)
    df["task"] = np.where(df["benchmark"] == "wmdp-bio", "wmdp-bio", df["benchmark"] + ":" + df["subtask"])
    df["chance"] = 1.0 / df["n_choices"]
    return df


def accuracy_table(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Main table. Unit of analysis is the QUESTION: with several option-order rotations, each
    question's score is its mean correctness over rotations, and the 95% CI comes from a
    bootstrap over questions (so correlated rotations do not shrink the interval). The
    one-sided binomial test against chance uses the first rotation only (independent items)."""
    rng = np.random.default_rng(seed)
    out = []
    for (model, template, task), g in df.groupby(["model", "template", "task"]):
        per_item = g.groupby("id")["correct"].mean().values
        n = len(per_item)
        boots = rng.choice(per_item, size=(n_boot, n), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        first = g[g["shift"] == g["shift"].min()]
        chance = float(first["chance"].mean())
        p = stats.binomtest(int(first["correct"].sum()), len(first), chance, alternative="greater").pvalue
        out.append({"model": model, "template": template, "task": task, "n_questions": n,
                    "accuracy": float(per_item.mean()), "ci_low": float(lo), "ci_high": float(hi),
                    "chance": chance, "above_chance_p": p})
    return pd.DataFrame(out).sort_values(["task", "template", "accuracy"], ascending=[True, True, False])


def position_bias_table(df: pd.DataFrame) -> pd.DataFrame:
    """Which letter slot does the model pick vs where the correct answer actually sits?
    With cyclic shifts the correct answer is uniformly distributed across slots, so a skew in
    predicted slots is pure position bias. Chi-square test against a uniform distribution."""
    out = []
    for (model, template, task), g in df.groupby(["model", "template", "task"]):
        n_opt = int(g["n_choices"].mode().iloc[0])
        counts = np.bincount(g["pred_pos"], minlength=n_opt)[:n_opt]
        expected = np.full(n_opt, counts.sum() / n_opt)
        chi2, p = stats.chisquare(counts, expected)
        row = {"model": model, "template": template, "task": task, "chi2": chi2, "p": p,
               "max_slot_share": counts.max() / counts.sum()}
        row.update({f"slot_{chr(65 + i)}": counts[i] / counts.sum() for i in range(n_opt)})
        out.append(row)
    return pd.DataFrame(out)


def order_robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per option-order rotation and prediction consistency across rotations."""
    out = []
    for (model, template, task), g in df.groupby(["model", "template", "task"]):
        if g["shift"].nunique() < 2:
            continue
        per_shift = g.groupby("shift")["correct"].mean()
        pivot = g.pivot_table(index="id", columns="shift", values="pred", aggfunc="first")
        consistent = (pivot.nunique(axis=1) == 1).mean()  # same ORIGINAL option chosen every time
        out.append({"model": model, "template": template, "task": task,
                    "acc_min": per_shift.min(), "acc_max": per_shift.max(),
                    "acc_range": per_shift.max() - per_shift.min(),
                    "consistency": consistent})
    return pd.DataFrame(out)


def calibration_table(df: pd.DataFrame, plot_dir: Path | None = None) -> pd.DataFrame:
    out = []
    for (model, template, task), g in df.groupby(["model", "template", "task"]):
        ece, rows = expected_calibration_error(g["conf"].values, g["correct"].values)
        out.append({"model": model, "template": template, "task": task, "ece": ece,
                    "mean_conf": g["conf"].mean(), "accuracy": g["correct"].mean()})
        if plot_dir is not None and template == "plain" and rows:
            fig, ax = plt.subplots(figsize=(3.6, 3.6))
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.plot([r["conf"] for r in rows], [r["acc"] for r in rows], "o-")
            ax.set(xlabel="confidence", ylabel="accuracy", title=f"{model.split('/')[-1]}\n{task} (ECE {ece:.3f})")
            fig.tight_layout()
            fig.savefig(plot_dir / f"calibration__{model.replace('/', '__')}__{task.replace(':', '_')}.png", dpi=130)
            plt.close(fig)
    return pd.DataFrame(out)


def template_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    acc = df.groupby(["model", "task", "template"])["correct"].mean().unstack("template")
    if acc.shape[1] < 2:
        return pd.DataFrame()
    acc["abs_diff"] = (acc.max(axis=1) - acc.min(axis=1))
    return acc.reset_index()


def agreement_and_difficulty(df: pd.DataFrame, template: str = "plain"):
    """Pairwise kappa between models' correctness vectors, and how many models solve each item."""
    d = df[(df["template"] == template) & (df["shift"] == 0)]
    kappas, hard = [], []
    for task, g in d.groupby("task"):
        mat = g.pivot_table(index="id", columns="model", values="correct", aggfunc="first").dropna()
        if mat.shape[1] < 2:
            continue
        models = list(mat.columns)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                kappas.append({"task": task, "model_a": models[i], "model_b": models[j],
                               "kappa": cohen_kappa(mat[models[i]].values, mat[models[j]].values)})
        solved = mat.sum(axis=1)
        hard.append({"task": task, "n_items": len(mat), "n_models": mat.shape[1],
                     "solved_by_none": float((solved == 0).mean()),
                     "solved_by_all": float((solved == mat.shape[1]).mean())})
    return pd.DataFrame(kappas), pd.DataFrame(hard)


def length_effect_table(df: pd.DataFrame, template: str = "plain") -> pd.DataFrame:
    """Accuracy by prompt-length quartile (token counts only; no question text is stored)."""
    out = []
    d = df[(df["template"] == template) & (df["shift"] == 0)]
    for (model, task), g in d.groupby(["model", "task"]):
        if len(g) < 40:
            continue
        g = g.assign(q=pd.qcut(g["prompt_tokens"], 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"], duplicates="drop"))
        for q, gg in g.groupby("q", observed=True):
            out.append({"model": model, "task": task, "length_quartile": str(q), "n": len(gg), "accuracy": gg["correct"].mean()})
    return pd.DataFrame(out)


def plot_main(acc: pd.DataFrame, out_dir: Path):
    a = acc[acc["template"] == "plain"] if (acc["template"] == "plain").any() else acc
    tasks = list(a["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4), sharey=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        t = a[a["task"] == task].sort_values("accuracy")
        y = np.arange(len(t))
        ax.barh(y, t["accuracy"], xerr=[t["accuracy"] - t["ci_low"], t["ci_high"] - t["accuracy"]], capsize=3)
        ax.axvline(t["chance"].mean(), color="k", ls="--", lw=1, label="chance")
        ax.set_yticks(y)
        ax.set_yticklabels([m.split("/")[-1] for m in t["model"]], fontsize=8)
        ax.set_title(task, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel("accuracy (95% bootstrap CI)")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_task.png", dpi=150)
    plt.close(fig)


def to_md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df is None or df.empty:
        return "_not available (needs more settings, e.g. --n-shifts 4 or --templates plain chat)_"
    d = df.copy()
    for c in d.select_dtypes("float").columns:
        d[c] = d[c].map(lambda x: format(x, floatfmt))
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.values]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    df = load_results(args.raw_dir)
    acc = accuracy_table(df)
    pos = position_bias_table(df)
    order = order_robustness_table(df)
    cal = calibration_table(df, out / "figures")
    tmpl = template_sensitivity_table(df)
    kappa, hard = agreement_and_difficulty(df)
    length = length_effect_table(df)
    plot_main(acc, out / "figures")

    for name, t in [("accuracy", acc), ("position_bias", pos), ("order_robustness", order),
                    ("calibration", cal), ("template_sensitivity", tmpl),
                    ("model_agreement", kappa), ("item_difficulty", hard), ("length_effect", length)]:
        if t is not None and not t.empty:
            t.to_csv(out / "tables" / f"{name}.csv", index=False)

    report = f"""# Results

Auto-generated by `python -m bioeval.analyze`. Items scored: {df['id'].nunique()} unique, {len(df)} predictions.

## Accuracy (mean over option-order rotations; 95% bootstrap CI over questions; one-sided binomial test vs chance)
{to_md(acc)}

![accuracy](figures/accuracy_by_task.png)

## Position bias (share of predictions per letter slot; chi-square vs uniform)
{to_md(pos)}

## Robustness to option order (needs `--n-shifts > 1`)
{to_md(order)}

## Calibration (ECE, 10 bins)
{to_md(cal)}

## Prompt-template sensitivity (needs `--templates plain chat`)
{to_md(tmpl)}

## Do models fail on the same questions? (Cohen's kappa of correctness)
{to_md(kappa)}

{to_md(hard)}

## Accuracy by prompt-length quartile
{to_md(length)}
"""
    (out / "REPORT.md").write_text(report)
    print(f"Wrote {out/'REPORT.md'}")


if __name__ == "__main__":
    main()
