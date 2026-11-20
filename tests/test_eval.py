"""Tests for the grader. Run all tests with:  uv run pytest"""

from ragkit.eval import Question, evaluate, is_hit, normalize


def test_normalize_strips_markdown_and_case():
    assert normalize("Uses the `text/event-stream` **Format**") == "uses the text/event-stream format"
    assert normalize("based on [HTTPX](https://x.org)") == "based on httpx"


def test_is_hit_ignores_formatting():
    assert is_hit("Set `response_model_exclude_unset=True` here", "response_model_exclude_unset=true")
    assert not is_hit("nothing relevant", "response_model_exclude_unset=true")


def test_recall_and_mrr_by_hand():
    qs = [Question("a", "q?", "apple", "x"), Question("b", "q?", "banana", "x")]
    # Question a: evidence at rank 2. Question b: never found.
    results = {"q?": ["cherry", "apple pie", "grape"]}
    res = evaluate(lambda q, k: results[q][:k], qs, ks=(1, 3))
    assert res.recall == {1: 0.0, 3: 0.5}  # a found within top-3, b never
    assert res.mrr == (1 / 2 + 0) / 2  # 0.25
    assert res.misses == ["b"]
