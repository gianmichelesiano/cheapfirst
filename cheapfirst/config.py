"""Caricamento e validazione della configurazione YAML."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import yaml


@dataclass
class ModelExtra:
    id: str
    provider: str
    name: str = ""
    tier: str = "cheap"
    context_length: int = 8192
    pricing: dict = field(default_factory=lambda: {"prompt_per_m": 0, "completion_per_m": 0})
    benchmarks: dict = field(default_factory=lambda: {"intelligence_index": 30, "coding_index": 30})


@dataclass
class RegistryConfig:
    auto_update: bool = True
    auto_update_interval_days: int = 7
    path: str = "~/.cheapfirst/registry.json"


@dataclass
class RoutingConfig:
    min_benchmark_score: float = 25.0
    mode: str = "cheapfirst"  # cheapfirst | balanced
    difficulty_thresholds: dict = field(default_factory=lambda: {"low": 0.33, "high": 0.70})
    verify: bool = True
    max_turns: int = 3
    skip_verify_confidence: float = 0.8
    max_cost_per_request: float = 0.05
    cost_weight: float = 0.3    # usato in modalità balanced
    quality_weight: float = 0.7 # usato in modalità balanced
    verify_cost_budget: float = 0.001
    unmeasured_policy: str = "exclude"


@dataclass
class MetricsConfig:
    enabled: bool = True
    db_path: str = "~/.cheapfirst/metrics.db"


@dataclass
class ServerConfig:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080


# Provider → base URL mapping per endpoint OpenAI-compatibile
PROVIDER_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "cohere": "https://api.cohere.com/v1",
    "perplexity": "https://api.perplexity.ai",
    "xai": "https://api.x.ai/v1",
    "github": "https://models.inference.ai.azure.com",
    "openrouter": "https://openrouter.ai/api/v1",
}

# Se un provider non è nella mappa, si assume sia locale o via proxy
# e si usa il valore della config provider_keys per quel provider come base_url


@dataclass
class CheapConfig:
    provider_keys: dict = field(default_factory=dict)
    models_extra: list = field(default_factory=list)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def resolve_provider_keys(self) -> dict:
        """Risolve le chiavi: se è ${VAR_NAME}, la prende dall'ambiente."""
        resolved = {}
        for name, value in self.provider_keys.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                env_val = os.environ.get(env_var)
                if env_val:
                    resolved[name] = env_val
            elif value:
                resolved[name] = value
        return resolved

    def resolve_path(self, p: str) -> Path:
        """Risolve ~ e variabili d'ambiente in un path."""
        return Path(os.path.expanduser(os.path.expandvars(p)))


def load_config(path: str | Path) -> CheapConfig:
    """Carica un file YAML e restituisce un oggetto CheapConfig."""
    path = Path(os.path.expanduser(str(path)))
    if not path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"File di configurazione vuoto: {path}")

    provider_keys = raw.get("provider_keys", {})

    models_extra = []
    for m in raw.get("models_extra", []):
        models_extra.append(ModelExtra(**m))

    registry_cfg = raw.get("registry", {})
    routing_cfg = raw.get("routing", {})
    metrics_cfg = raw.get("metrics", {})
    server_cfg = raw.get("server", {})

    return CheapConfig(
        provider_keys=provider_keys,
        models_extra=models_extra,
        registry=RegistryConfig(**registry_cfg),
        routing=RoutingConfig(**routing_cfg),
        metrics=MetricsConfig(**metrics_cfg),
        server=ServerConfig(**server_cfg),
    )
