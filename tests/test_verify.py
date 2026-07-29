"""Test del verify."""

from cheapfirst.verify import verify_response, Verdict
from cheapfirst.classifier import TaskSignature


def test_verify_accepts_valid_code():
    resp = {"text": "```python\ndef hello():\n    pass\n```"}
    sig = TaskSignature(task="code", difficulty=0.5, confidence=0.7)
    assert verify_response(resp, sig) == Verdict.ACCEPT


def test_verify_revises_empty_code():
    resp = {"text": "non lo so"}
    sig = TaskSignature(task="code", difficulty=0.5, confidence=0.7)
    assert verify_response(resp, sig) == Verdict.REVISE


def test_verify_accepts_translation():
    resp = {"text": "Ciao"}
    sig = TaskSignature(task="translation", difficulty=0.2, confidence=0.9)
    assert verify_response(resp, sig) == Verdict.ACCEPT


def test_verify_revises_empty():
    resp = {"text": ""}
    sig = TaskSignature(task="general", difficulty=0.3, confidence=0.5)
    assert verify_response(resp, sig) == Verdict.REVISE


def test_verify_detects_unbalanced_brackets():
    resp = {"text": "function test() { return { }"}
    sig = TaskSignature(task="code", difficulty=0.5, confidence=0.7)
    assert verify_response(resp, sig) == Verdict.REVISE  # { count != } count


def test_verify_math_with_numbers():
    resp = {"text": "Il risultato è 42"}
    sig = TaskSignature(task="math", difficulty=0.6, confidence=0.7)
    assert verify_response(resp, sig) == Verdict.ACCEPT


def test_verify_math_too_vague():
    resp = {"text": "non ho capito"}
    sig = TaskSignature(task="math", difficulty=0.6, confidence=0.7)
    assert verify_response(resp, sig) == Verdict.REVISE
