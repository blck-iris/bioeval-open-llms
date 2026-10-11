# bioeval-open-llms

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blck-iris/bioeval-open-llms/blob/main/notebooks/colab_run.ipynb)

A small, reproducible harness for measuring how well **open-weight language models** answer
**public biology and biosecurity multiple-choice benchmarks**, with a focus on *how trustworthy the measurement is*,
not just the headline accuracy.

Benchmarks (both public):

| Benchmark | What it measures | Source |
|---|---|---|
| **WMDP-Bio** | Hazardous-knowledge *proxy* questions in biosecurity (1,273 four-way multiple-choice) | Li et al., 2024, `cais/wmdp` |
| **LAB-Bench** (text-only subtasks: LitQA2, SeqQA, CloningScenarios, ProtocolQA, DbQA) | Practical biology research skills | Laurent et al., 2024, `futurehouse/lab-bench` |

## Why this exists

Single-number benchmark scores hide several problems that matter when benchmarks are used to track
model capability or risk. This project quantifies them:

1. **Uncertainty** - 95% bootstrap confidence intervals over questions, and a test against the chance baseline.
2. **Position bias** - models often favour a letter. Every question is shown in several cyclic option orders
   so the correct answer lands in each slot equally often; the analysis reports the bias and how much accuracy moves.
3. **Order consistency** - what fraction of questions get the *same* answer under every option order.
4. **Prompt sensitivity** - plain completion prompt vs. chat template.
5. **Calibration** - expected calibration error of the letter probabilities.
6. **Failure structure** - do different models miss the same questions? Is accuracy tied to prompt length?

## How scoring works

For each question we build a prompt ending at "Answer:", run **one forward pass**, and compare the next-token probabilities of the option letters (A, B, C, ...).
There is **no free-form text generation**, which makes the evaluation fast, deterministic and cheap, and it works for base and instruct models.
Predictions are mapped back to the original option indices before scoring.

Saved outputs contain item ids, predicted option, probabilities and token counts only. **Question text is never written to results files.**

## Quickstart (Google Colab, free T4)

1. Open the notebook with the badge above (or upload `notebooks/colab_run.ipynb`), set the runtime to **T4 GPU**.
2. Run all cells. A 2-minute smoke test runs first; the full run takes roughly 1.5 to 3 hours.
3. Results land in `results/` (`REPORT.md`, `tables/*.csv`, `figures/*.png`).

### Local / command line

```bash
git clone https://github.com/blck-iris/bioeval-open-llms.git && cd bioeval-open-llms
pip install -r requirements.txt
python -m pytest -q                       # unit tests + end-to-end analysis test on synthetic data

# smoke test
python -m bioeval.run_eval --models Qwen/Qwen2.5-0.5B-Instruct --benchmarks wmdp-bio --limit 50 --out-dir results/smoke

# full run
python -m bioeval.run_eval \
  --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct Qwen/Qwen2.5-3B-Instruct \
  --benchmarks wmdp-bio labbench:LitQA2 labbench:SeqQA labbench:CloningScenarios \
  --templates plain chat --n-shifts 4

python -m bioeval.analyze --raw-dir results/raw --out-dir results
```

Useful flags: `--limit N` (items per benchmark), `--load-in-4bit` (7B models on a T4), `--add-insufficient` (append LAB-Bench's
"Insufficient information" option), `--batch-size`, `--overwrite`. Existing per-model result files are skipped, so interrupted runs resume.

## Results

Four open models (Qwen2.5-0.5B/1.5B/3B-Instruct, SmolLM2-1.7B-Instruct), 2,105 questions, 2 prompt templates, 4 cyclic option orders (67,360 predictions).
Run on a Colab T4. Full tables and figures: [`results/REPORT.md`](results/REPORT.md).

**Accuracy, plain prompt, averaged over the 4 option orders** (95% bootstrap CI over questions in brackets; chance is about 25%):

| Model | wmdp-bio (n=1273) | LAB-Bench LitQA2 (n=199) | LAB-Bench SeqQA (n=600) | LAB-Bench CloningScenarios (n=33) |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 56.2% [54.0, 58.3] | 29.8% [25.6, 34.0] | 38.1% [36.0, 40.5] | 21.2% [14.4, 28.8] |
| Qwen2.5-1.5B-Instruct | 66.7% [64.3, 68.9] | 34.4% [29.5, 39.3] | 37.5% [35.2, 40.0] | 22.0% [14.4, 29.5] |
| Qwen2.5-3B-Instruct | 70.0% [67.8, 72.2] | 40.6% [35.4, 46.2] | 42.2% [39.4, 44.8] | 22.0% [13.6, 31.1] |
| SmolLM2-1.7B-Instruct | 50.9% [48.9, 53.0] | 26.9% [23.0, 30.9] | 27.0% [25.5, 28.4] | 23.5% [16.7, 31.1] |

![accuracy](results/figures/accuracy_by_task.png)

**What the run shows**

- **WMDP-Bio is well above chance, LAB-Bench mostly is not.** All four models beat chance on WMDP-Bio (50.9% to 70.0%). On LAB-Bench they are far weaker:
  the best SeqQA and LitQA2 scores are 42.2% and 40.6%, and no model is distinguishable from chance on CloningScenarios (only 33 questions, so intervals are wide).
- **Position bias is large and nearly universal.** 30 of 32 model/template/task combinations show a significant skew in which letter the model picks.
  Under the plain prompt, SmolLM2-1.7B picks slot A for 45.1% of WMDP-Bio answers (25% expected), and Qwen2.5-0.5B and 1.5B pick A for about 50% of SeqQA answers.
- **Answers are not stable under option reordering.** The same original option is chosen under all four orders for only 33% to 71% of WMDP-Bio questions
  (Qwen2.5-3B highest at 71.2%) and for 1.7% to 17.7% of SeqQA questions with the plain prompt. Headline accuracy moves little on WMDP-Bio (under 3.2 points across orders), which hides how often individual answers flip.
- **Models are overconfident.** Qwen2.5-3B reaches 70.0% on WMDP-Bio with 90.6% mean confidence (ECE 0.21); on LitQA2 with the chat prompt it has 36.9% accuracy at 83.5% mean confidence.
  Confidence rises with model size faster than accuracy does.
- **Prompt format matters, mostly on LAB-Bench.** The plain prompt beats the chat template by up to 7.2 points on SeqQA (Qwen2.5-3B) and by 1 to 4 points on WMDP-Bio,
  but for SmolLM2-1.7B on SeqQA the chat template is 5 points better. No single format is best.
- **Models fail on different questions.** Pairwise Cohen's kappa of correctness is 0.37 to 0.57 on WMDP-Bio and lower on LAB-Bench. 15.9% of WMDP-Bio questions are solved by none of the four models and 36.3% by all four;
  on LAB-Bench, 30% to 45% of questions are solved by none.

Take-away: single-number scores from small open models on multiple-choice biology benchmarks are sensitive to prompt format and option order, and are poorly calibrated, so any capability claim needs the controls reported here.

## Repository layout

```
bioeval/
  data.py        dataset loaders (WMDP-Bio, LAB-Bench) with a common schema
  scoring.py     prompt building + log-likelihood multiple-choice scoring
  metrics.py     Wilson/ECE/kappa/permutation helpers (pure numpy, unit tested)
  run_eval.py    CLI: run models x benchmarks x templates x option orders
  analyze.py     tables, figures and REPORT.md
tests/           unit tests and an end-to-end analysis test on synthetic results
notebooks/       Colab notebook
results/         raw per-model jsonl, tables, figures, REPORT.md
```

## Limitations

- Multiple-choice letter likelihood measures *knowledge recognition on these benchmarks*, not open-ended capability, agentic ability, or real-world uplift.
  A model scoring well here is not evidence about what it could do for a person in practice, and a low score is not evidence of safety.
- Only small open models that fit a free GPU are evaluated by default. Conclusions do not transfer to frontier models.
- LAB-Bench is evaluated closed-book on text-only subtasks; figure/table/supplement subtasks and tool use are out of scope.
- Benchmarks can be contaminated or noisy; results depend on the prompt template (see the template-sensitivity table).
- Dataset column names follow the public dataset cards; if a revision changes them, edit `bioeval/data.py`.

## Responsible use

This project only runs public benchmarks and compares answer-letter probabilities. It does not generate, store or distribute
hazardous content, and it saves no question text. Please keep forks and extensions in that spirit.

## Citation and credits

- Li et al., *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning*, 2024.
- Laurent et al., *LAB-Bench: Measuring Capabilities of Language Models for Biology Research*, 2024.
- Model weights belong to their respective authors (Qwen, Hugging Face SmolLM2); check each model card for its license.

MIT License (code only).
#
