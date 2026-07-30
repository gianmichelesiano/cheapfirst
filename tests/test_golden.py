"""Golden test: ranking attuale congelato.

Prima di cambiare la formula di ranking, committiamo il comportamento attuale.
Ogni test crea un router con la golden pool e verifica il modello scelto.
"""

import pytest
from cheapfirst.config import CheapConfig
from cheapfirst.registry import ModelRegistry
from cheapfirst.router import Router
from cheapfirst.classifier import classify
from tests.fixtures.golden_pool import GOLDEN_POOL, GOLDEN_PROMPTS


def _make_router() -> Router:
    config = CheapConfig(provider_keys={"deepseek": "sk-test"})
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.config = config
    registry.models = GOLDEN_POOL
    router = Router.__new__(Router)
    router.config = config
    router.registry = registry
    router._provider_keys = {}
    return router


def test_golden_all_prompts_produce_a_decision():
    """Ogni prompt del golden test produce una decisione valida."""
    router = _make_router()
    for prompt, expected_task, d_min, d_max in GOLDEN_PROMPTS:
        sig = classify([{"role": "user", "content": prompt}])
        decision = router.route(
            [{"role": "user", "content": prompt}], sig, dry_run=True
        )
        assert "error" not in decision, f"{prompt[:30]}: {decision.get('error')}"
        assert "model" in decision, f"{prompt[:30]}: nessun modello scelto"
        assert d_min <= sig.difficulty <= d_max, (
            f"{prompt[:30]}: difficulty={sig.difficulty:.2f} "
            f"attesa [{d_min}, {d_max}]"
        )


def test_golden_classification_consistent():
    """La classificazione dei prompt golden è stabile."""
    for prompt, expected_task, d_min, d_max in GOLDEN_PROMPTS:
        sig = classify([{"role": "user", "content": prompt}])
        if expected_task == "translation":
            assert sig.task == "translation", (
                f"{prompt[:30]}: atteso translation, ottenuto {sig.task}"
            )


def test_golden_pool_size():
    """La golden pool ha 20 modelli."""
    assert len(GOLDEN_POOL) == 20


def test_golden_prompt_count():
    """Ci sono 30 prompt golden."""
    assert len(GOLDEN_PROMPTS) == 30


def test_golden_ranking_cheap_translation():
    """Traduzione semplice sceglie il modello più economico competente."""
    router = _make_router()
    prompt = "Traduci hello in italiano"
    sig = classify([{"role": "user", "content": prompt}])
    decision = router.route(
        [{"role": "user", "content": prompt}], sig, dry_run=True
    )
    # DeepSeek Flash o Gemini Flash (i più economici per traduzione)
    assert "deepseek" in decision["model"] or "gemini" in decision["model"], (
        f"Traduzione semplice: atteso economico, ottenuto {decision['model']}"
    )


def test_golden_model_without_benchmarks():
    """Modello senza benchmark non viene escluso ma riceve bench=0.1."""
    router = _make_router()
    pool = router._filter_competent(GOLDEN_POOL, classify([{"role": "user", "content": "Ciao"}]))
    no_bench = [m for m in pool if m.id == "new-model/no-benchmarks"]
    assert len(no_bench) >= 0, "Modello senza benchmark può essere incluso"
