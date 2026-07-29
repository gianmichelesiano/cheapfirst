"""Test del registry modelli."""

import json
import tempfile
from pathlib import Path

from cheapfirst.config import CheapConfig, ModelExtra
from cheapfirst.registry import ModelRegistry, ModelSpec


def _make_registry(config) -> ModelRegistry:
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.config = config
    registry.models = []
    return registry


def test_registry_merge_custom():
    """I modelli custom dal config YAML vengono aggiunti correttamente."""
    config = CheapConfig(
        provider_keys={"deepseek": "sk-test"},
        models_extra=[
            ModelExtra(
                id="ollama/llama3.2",
                provider="local",
                tier="free",
                pricing={"prompt_per_m": 0, "completion_per_m": 0},
                benchmarks={"intelligence_index": 30, "coding_index": 35},
            ),
        ],
    )
    registry = _make_registry(config)
    registry._merge_custom()

    assert len(registry.models) == 1
    assert registry.models[0].id == "ollama/llama3.2"
    assert registry.models[0].pricing["completion_per_m"] == 0


def test_registry_active_pool():
    """Solo i modelli con provider attivo vengono restituiti."""
    config = CheapConfig(provider_keys={"deepseek": "sk-test"})
    registry = _make_registry(config)
    registry.models = [
        ModelSpec(id="deepseek/test", provider="deepseek"),
        ModelSpec(id="anthropic/test", provider="anthropic"),
    ]

    pool = registry.get_active_pool({"deepseek": "sk-test"})
    assert len(pool) == 1
    assert pool[0].provider == "deepseek"


def test_registry_save_and_load():
    """Salvataggio e caricamento del registry funzionano."""
    config = CheapConfig(provider_keys={"deepseek": "sk-test"})
    registry = _make_registry(config)
    registry.models = [
        ModelSpec(
            id="deepseek/test",
            provider="deepseek",
            name="Test Model",
            tier="cheap",
            pricing={"prompt_per_m": 0.15, "completion_per_m": 0.60},
            benchmarks={"intelligence_index": 40},
        ),
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        pass  # crea file temporaneo
    temp_path = Path(f.name)

    # Salva
    config = CheapConfig(provider_keys={"deepseek": "sk-test"})
    registry = _make_registry(config)
    registry.models = [
        ModelSpec(
            id="deepseek/test",
            provider="deepseek",
            name="Test Model",
            tier="cheap",
            pricing={"prompt_per_m": 0.15, "completion_per_m": 0.60},
            benchmarks={"intelligence_index": 40},
        ),
    ]
    # Usa il percorso temporaneo per il salvataggio
    original_path = registry.config.registry.path
    registry.config.registry.path = str(temp_path)
    registry._save()

    # Carica da file
    registry2 = _make_registry(config)
    registry2.config.registry.path = str(temp_path)
    registry2._load_from_file(temp_path)
    assert len(registry2.models) == 1
    assert registry2.models[0].id == "deepseek/test"

    Path(temp_path).unlink(missing_ok=True)


def test_model_spec_defaults():
    """ModelSpec ha valori di default sensati."""
    spec = ModelSpec(id="test/model", provider="test")
    assert spec.tier == "cheap"
    assert spec.context_length == 8192
    assert spec.pricing["completion_per_m"] == 0
    assert spec.benchmarks["intelligence_index"] == 0
    assert spec.strength == 0.5


def test_registry_merge_skips_duplicates():
    """Modelli custom con ID già presente non vengono duplicati."""
    config = CheapConfig(
        provider_keys={"deepseek": "sk-test"},
        models_extra=[
            ModelExtra(id="deepseek/v4-flash", provider="deepseek"),
        ],
    )
    registry = _make_registry(config)
    registry.models = [
        ModelSpec(id="deepseek/v4-flash", provider="deepseek"),
    ]
    registry._merge_custom()

    assert len(registry.models) == 1  # non duplicato
