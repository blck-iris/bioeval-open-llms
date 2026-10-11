#!/usr/bin/env bash
# Publish the project to GitHub (run from the repo root). Requires git and the GitHub CLI (gh), or create the repo in the web UI.
set -euo pipefail
git init -b main
git add .
git commit -m "bioeval: open-LLM evaluation on WMDP-Bio and LAB-Bench"
gh repo create blck-iris/bioeval-open-llms --public --source=. --remote=origin --push \
  --description "Reproducible evaluation of open LLMs on public biology/biosecurity benchmarks (WMDP-Bio, LAB-Bench)"
