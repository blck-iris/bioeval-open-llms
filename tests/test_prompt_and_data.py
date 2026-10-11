from bioeval.data import _stable_rng
from bioeval.scoring import build_prompt


def test_plain_prompt_format():
    p = build_prompt("What is 2+2?", ["1", "4", "5", "9"], "math", "plain")
    assert p.startswith("The following is a multiple choice question about math.")
    assert "A. 1" in p and "B. 4" in p and p.rstrip().endswith("Answer:")


def test_stable_rng_is_deterministic():
    a = _stable_rng("q1", 0).random()
    b = _stable_rng("q1", 0).random()
    c = _stable_rng("q2", 0).random()
    assert a == b and a != c
