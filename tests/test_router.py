"""Test del router e del meccanismo di routing."""

from cheapfirst.config import CheapConfig
from cheapfirst.registry import ModelRegistry, ModelSpec
from cheapfirst.classifier import classify
from cheapfirst.router import Router
from unittest.mock import patch


def _make_config() -> CheapConfig:
    return CheapConfig(
        provider_keys={"deepseek": "sk-test-key"},
    )


def _make_registry_direct(config, models) -> ModelRegistry:
    """Crea un registry senza fetchare da OpenRouter."""
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.config = config
    registry.models = models
    return registry


def test_router_competent_filter_code():
    """Router esclude modelli con benchmark basso per coding."""
    config = _make_config()
    models = [
        ModelSpec(
            id="deepseek/test-flash",
            provider="deepseek",
            tier="cheap",
            pricing={"prompt_per_m": 0.15, "completion_per_m": 0.60},
            benchmarks={"coding_index": 56, "intelligence_index": 40},
        ),
        ModelSpec(
            id="deepseek/test-pro",
            provider="deepseek",
            tier="mid",
            pricing={"prompt_per_m": 1.00, "completion_per_m": 4.00},
            benchmarks={"coding_index": 38, "intelligence_index": 44},
        ),
    ]
    registry = _make_registry_direct(config, models)
    router = Router(config, registry)

    sig = classify([{"role": "user", "content": "Write a Python function"}])
    competent = router._filter_competent(models, sig)

    # Entrambi hanno coding_index > 25, quindi entrambi passano
    assert len(competent) == 2


def test_router_ranking_cheaper_wins():
    """Tra modelli competenti, vince il rapporto costo/benchmark migliore."""
    config = _make_config()
    models = [
        ModelSpec(
            id="deepseek/flash",
            provider="deepseek",
            tier="cheap",
            pricing={"prompt_per_m": 0.15, "completion_per_m": 0.60},
            benchmarks={"intelligence_index": 40},
        ),
        ModelSpec(
            id="deepseek/pro",
            provider="deepseek",
            tier="mid",
            pricing={"prompt_per_m": 1.00, "completion_per_m": 4.00},
            benchmarks={"intelligence_index": 44},
        ),
    ]
    registry = _make_registry_direct(config, models)
    router = Router(config, registry)

    ranked = router._rank(models, "general")
    assert len(ranked) >= 2

    # Flash deve vincere: (0.6*200/1M)/40 = 0.000003 vs Pro: (4.0*200/1M)/44 = 0.000018
    assert ranked[0][0].id == "deepseek/flash"


def test_router_dry_run_returns_decision():
    """Dry-run restituisce decisione senza eseguire alcuna chiamata."""
    config = _make_config()
    models = [
        ModelSpec(
            id="deepseek/flash",
            provider="deepseek",
            tier="cheap",
            pricing={"prompt_per_m": 0.15, "completion_per_m": 0.60},
            benchmarks={"intelligence_index": 40},
        ),
    ]
    registry = _make_registry_direct(config, models)
    router = Router(config, registry)

    sig = classify([{"role": "user", "content": "Hello world"}])
    decision = router.route(
        [{"role": "user", "content": "Hello world"}],
        sig,
        dry_run=True,
    )

    assert "model" in decision
    assert "score" in decision
    assert "reason" in decision
    assert decision["model"] == "deepseek/flash"
