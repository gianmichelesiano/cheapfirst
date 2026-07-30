"""Test del router e del meccanismo di routing."""

from cheapfirst.config import CheapConfig
from cheapfirst.registry import ModelRegistry, ModelSpec
from cheapfirst.classifier import classify
from cheapfirst.router import Router
from unittest.mock import patch


def _make_config() -> CheapConfig:
    return CheapConfig(
        provider_keys={"deepseek": "sk-test-key", "vendor": "sk-test-key"},
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

    ranked, _ = router._rank(models, "general")
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


def test_router_rank_handles_none_benchmarks():
    """_rank non crasha con benchmark=None quando _filter_competent svuota il pool.

    Riproduce TypeError: '<=' not supported between instances of 'NoneType' and 'int'.
    """
    config = _make_config()
    models = [
        ModelSpec(
            id="vendor/unmeasured-a",
            provider="vendor",
            tier="cheap",
            pricing={"prompt_per_m": 0.15, "completion_per_m": 0.60},
            benchmarks={"intelligence_index": None, "coding_index": None, "agentic_index": None},
        ),
        ModelSpec(
            id="vendor/unmeasured-b",
            provider="vendor",
            tier="mid",
            pricing={"prompt_per_m": 1.00, "completion_per_m": 4.00},
            benchmarks={"intelligence_index": None, "coding_index": None, "agentic_index": None},
        ),
    ]
    registry = _make_registry_direct(config, models)
    router = Router(config, registry)

    # _filter_competent scarta tutti (tutti None), fallback = pool (tutti None)
    ranked, excluded = router._rank(models, "general")

    # Con exclude: nessun modello ranked, entrambi esclusi
    assert len(ranked) == 0, "tutti i modelli con benchmark=None devono essere esclusi da _rank"
    assert excluded == 2, f"devono essere esclusi 2 modelli, esclusi {excluded}"


def test_router_dry_run_with_none_benchmarks_shows_exclusion():
    """Dry-run con modelli None nel fallback non crasha e menziona l'esclusione."""
    config = _make_config()
    # Tutti i modelli con benchmark None — _filter_competent scarta tutti,
    # fallback = pool, _rank li esclude e riporta il conteggio
    models = [
        ModelSpec(
            id="vendor/unmeasured-a",
            provider="vendor",
            tier="cheap",
            pricing={"prompt_per_m": 0.15, "completion_per_m": 0.60},
            benchmarks={"intelligence_index": None, "coding_index": None, "agentic_index": None},
        ),
        ModelSpec(
            id="vendor/unmeasured-b",
            provider="vendor",
            tier="mid",
            pricing={"prompt_per_m": 1.00, "completion_per_m": 4.00},
            benchmarks={"intelligence_index": None, "coding_index": None, "agentic_index": None},
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

    # Non deve crashare — anche se nessun modello ha benchmark
    assert "error" in decision
    assert "Nessun modello disponibile dopo il ranking" in decision["error"]


def test_router_mixed_benchmarks_exclude_some():
    """Con pool misto (alcuni None, alcuni normali), i None vengono esclusi e contati.

    Scenario: _filter_competent fa passare entrambi (stesso tier, None non è
    incompetente), _rank esclude il None e ranka il misurato.
    """
    config = _make_config()
    models = [
        ModelSpec(
            id="vendor/measured",
            provider="vendor",
            tier="cheap",
            pricing={"prompt_per_m": 0.15, "completion_per_m": 0.60},
            benchmarks={"intelligence_index": 40},
        ),
        ModelSpec(
            id="vendor/unmeasured",
            provider="vendor",
            tier="cheap",
            pricing={"prompt_per_m": 0.15, "completion_per_m": 0.60},
            benchmarks={"intelligence_index": None},
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

    assert decision["model"] == "vendor/measured"
    reason = decision["reason"]
    assert "1 modelli esclusi" in reason, f"reason deve menzionare esclusione: {reason}"


def test_router_unmeasured_impute_from_tier():
    """Con unmeasured_policy=impute_from_tier, i None ottengono un default per tier."""
    config = CheapConfig(
        provider_keys={"vendor": "sk-test-key"},
    )
    config.routing.unmeasured_policy = "impute_from_tier"
    models = [
        ModelSpec(
            id="vendor/cheap-model",
            provider="vendor",
            tier="cheap",
            pricing={"prompt_per_m": 0.15, "completion_per_m": 0.60},
            benchmarks={"intelligence_index": None},
        ),
        ModelSpec(
            id="vendor/ultra-model",
            provider="vendor",
            tier="ultra",
            pricing={"prompt_per_m": 15.00, "completion_per_m": 60.00},
            benchmarks={"intelligence_index": None},
        ),
    ]
    registry = _make_registry_direct(config, models)
    router = Router(config, registry)

    # Con impute_from_tier, nessuno escluso — entrambi ricevono default dal tier
    ranked, excluded = router._rank(models, "general")
    assert excluded == 0, "con impute_from_tier nessun modello deve essere escluso"

    # cheap ha bench=30, ultra ha bench=70
    cheap_score = ranked[0][1]  # cost/bench, più basso = meglio
    ultra_score = ranked[1][1]
    # cheap deve essere più economico perché ha prezzo molto più basso
    assert cheap_score < ultra_score, "il modello cheap deve avere punteggio migliore"
    # Verifica che i benchmark siano state imputati
    _, _, _, cheap_bench = ranked[0]
    _, _, _, ultra_bench = ranked[1]
    assert cheap_bench == 30, f"cheap deve avere bench=30, ha {cheap_bench}"
    assert ultra_bench == 70, f"ultra deve avere bench=70, ha {ultra_bench}"
