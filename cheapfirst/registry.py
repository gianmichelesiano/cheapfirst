"""Gestione del registry dei modelli.

Scarica da OpenRouter API, fa merge con modelli custom YAML.
"""

import json
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from dataclasses import dataclass, field, asdict


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
USER_AGENT = "cheapfirst/0.1.0"


@dataclass
class ModelSpec:
    id: str
    provider: str
    name: str = ""
    tier: str = "cheap"
    context_length: int = 8192
    modality: str = "text"
    pricing: dict = field(default_factory=lambda: {
        "prompt_per_m": 0,
        "completion_per_m": 0,
        "cache_read_per_m": 0,
    })
    benchmarks: dict = field(default_factory=lambda: {
        "intelligence_index": 0,
        "coding_index": 0,
        "agentic_index": 0,
    })
    strength: float = 0.5  # forza complessiva del modello (0-1)


class ModelRegistry:
    """Contiene e gestisce il pool di modelli."""

    def __init__(self, config):
        self.config = config
        self.models: list[ModelSpec] = []
        self._load()

    def _load(self):
        """Carica il registry: prima cercalo locale, poi fetch remoto."""
        registry_path = Path(self.config.resolve_path(self.config.registry.path))
        if registry_path.exists():
            age = time.time() - registry_path.stat().st_mtime
            age_days = age / 86400
            if age_days < self.config.registry.auto_update_interval_days:
                self._load_from_file(registry_path)
                self._merge_custom()
                return

        # Fetch remoto
        self.update()

    def update(self):
        """Scarica l'ultimo registry da OpenRouter, salva localmente."""
        data = self._fetch_openrouter()
        if data:
            self._parse_openrouter(data)
            self._merge_custom()
            self._save()
        else:
            # Fallback: carica da file se esiste
            registry_path = Path(self.config.resolve_path(self.config.registry.path))
            if registry_path.exists():
                self._load_from_file(registry_path)
                self._merge_custom()

    def _fetch_openrouter(self) -> Optional[dict]:
        """Chiama OpenRouter API e restituisce i dati JSON."""
        try:
            req = Request(OPENROUTER_MODELS_URL, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except Exception as e:
            print(f"Attenzione: impossibile scaricare registry da OpenRouter: {e}")
            return None

    def _parse_openrouter(self, data: dict):
        """Converte la risposta OpenRouter in ModelSpec list."""
        self.models = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            if not mid or "/" not in mid:
                continue

            provider = mid.split("/")[0]
            pricing = m.get("pricing", {})
            benchmarks = m.get("benchmarks", {})
            aa = benchmarks.get("artificial_analysis", {})

            # Prezzo per milione di token
            prompt_raw = float(pricing.get("prompt", 0))
            completion_raw = float(pricing.get("completion", 0))

            # Assegna tier in base al prezzo
            completion_per_m = completion_raw * 1_000_000
            if completion_per_m <= 1.0:
                tier = "cheap"
            elif completion_per_m <= 5.0:
                tier = "mid"
            elif completion_per_m <= 20.0:
                tier = "frontier"
            else:
                tier = "ultra"

            # Forza tier per modelli free
            if completion_per_m == 0 and "free" in mid:
                tier = "free"

            spec = ModelSpec(
                id=mid,
                provider=provider,
                name=m.get("name", mid),
                tier=tier,
                context_length=m.get("context_length", 4096),
                modality=m.get("architecture", {}).get("modality", "text"),
                pricing={
                    "prompt_per_m": round(prompt_raw * 1_000_000, 4),
                    "completion_per_m": round(completion_per_m, 4),
                    "cache_read_per_m": round(
                        float(pricing.get("input_cache_read", 0)) * 1_000_000, 4
                    ),
                },
                benchmarks={
                    "intelligence_index": aa.get("intelligence_index", 0),
                    "coding_index": aa.get("coding_index", 0),
                    "agentic_index": aa.get("agentic_index", 0),
                },
                strength=round(min(1.0, (aa.get("intelligence_index") or 0) / 100), 2),
            )
            self.models.append(spec)

    def _merge_custom(self):
        """Aggiunge i modelli custom dal config YAML."""
        existing_ids = {m.id for m in self.models}
        for extra in self.config.models_extra:
            if extra.id not in existing_ids:
                spec = ModelSpec(
                    id=extra.id,
                    provider=extra.provider,
                    name=extra.id,
                    tier=extra.tier,
                    context_length=extra.context_length,
                    pricing=extra.pricing,
                    benchmarks=extra.benchmarks,
                    strength=min(1.0, extra.benchmarks.get("intelligence_index", 30) / 100),
                )
                self.models.append(spec)

    def _save(self):
        """Salva il registry su disco."""
        path = Path(self.config.resolve_path(self.config.registry.path))
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "openrouter",
            "models": [asdict(m) for m in self.models],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_from_file(self, path: Path):
        """Carica il registry da file locale."""
        with open(path) as f:
            data = json.load(f)
        for m in data.get("models", []):
            self.models.append(ModelSpec(**m))

    def get_active_pool(self, provider_keys: dict) -> list[ModelSpec]:
        """Restituisce solo i modelli i cui provider hanno API key configurata."""
        active_providers = set(provider_keys.keys())
        return [m for m in self.models if m.provider in active_providers]

    def get_models_by_tier(self, pool: list[ModelSpec], tier: str) -> list[ModelSpec]:
        """Filtra il pool per tier."""
        return [m for m in pool if m.tier == tier or m.tier == "free"]

    def status(self) -> dict:
        return {
            "models_count": len(self.models),
            "last_update": "sconosciuto",
            "age_days": 0,
        }
