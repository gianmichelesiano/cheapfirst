"""Test del classificatore euristico."""

from cheapfirst.classifier import classify, TaskSignature


def test_classify_translation():
    sig = classify([{"role": "user", "content": "Translate hello to Italian"}])
    assert sig.task == "translation"
    assert sig.difficulty < 0.4
    assert sig.confidence > 0.7


def test_classify_code():
    sig = classify([{"role": "user", "content": "Write a Python function to sort a list"}])
    assert sig.task == "code"
    assert sig.confidence > 0.7


def test_classify_code_with_backticks():
    sig = classify([{"role": "user", "content": "```\ndef hello():\n    pass\n```"}])
    assert sig.task == "code"
    assert sig.confidence > 0.8


def test_classify_math():
    sig = classify([{"role": "user", "content": "Solve the integral of x^2 dx"}])
    assert sig.task == "math"
    assert sig.difficulty > 0.3


def test_classify_factual():
    sig = classify([{"role": "user", "content": "What is the capital of France?"}])
    assert sig.task == "factual"
    assert sig.difficulty < 0.4


def test_classify_complex():
    sig = classify([{
        "role": "user",
        "content": "Design a distributed rate limiting system with eventual consistency"
    }])
    # "Design" attiva il pattern creative, ma il task è tecnico
    assert sig.difficulty > 0.5  # complesso
    assert sig.confidence > 0.5


def test_classify_general():
    sig = classify([{"role": "user", "content": "Ciao, come stai?"}])
    assert sig.task == "general"
    assert sig.difficulty < 0.4


def test_classify_sensitive():
    sig = classify([{"role": "user", "content": "My password is secret123"}])
    assert sig.sensitive is True


def test_classify_creative():
    sig = classify([{"role": "user", "content": "Brainstorm ideas for a new mobile app"}])
    assert sig.task == "creative"


def test_classify_empty():
    sig = classify([])
    assert sig.task == "general"
    assert sig.confidence == 0.3


def test_classify_italian_text():
    """Task semplice in italiano."""
    sig = classify([{"role": "user", "content": "Traduci ciao in inglese"}])
    assert sig.task == "translation"
    assert sig.difficulty < 0.4


def test_classify_long_complex():
    """Testo lungo con richiesta complessa."""
    sig = classify([{
        "role": "user",
        "content": "Analyze the trade-offs between microservices and monolith architectures "
                   "for a high-traffic e-commerce platform. Consider deployment complexity, "
                   "scalability, team organization, and operational costs. Provide a "
                   "step-by-step decision framework with formal proof of your reasoning."
    }])
    assert sig.difficulty > 0.7
    assert sig.confidence > 0.7
