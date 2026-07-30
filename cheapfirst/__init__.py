"""cheapfirst — LLM router: prova il cheap, verifica, scala.

Risparmia fino all'80% sui costi API LLM senza sacrificare qualità.
Classifica il task, ranka i modelli per costo/benchmark, esegue e verifica.
"""

from .config import CheapConfig, load_config
from .classifier import TaskSignature, classify
from .router import Router
from .registry import ModelRegistry
from .executor import execute
from .verify import verify_response, Verdict
from .metrics import MetricsLogger

from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_PATHS = [
    Path("cheapfirst.yaml"),
    Path("cheapfirst.yml"),
    Path.home() / ".cheapfirst" / "config.yaml",
]


class CheapFirst:
    """Punto d'ingresso principale per cheapfirst."""

    def __init__(self, config: Optional[str | dict | Path] = None):
        if config is None:
            config = self._find_config()
        if isinstance(config, (str, Path)):
            self.config = load_config(config)
        elif isinstance(config, dict):
            self.config = CheapConfig(**config)
        else:
            raise TypeError(f"config deve essere path, dict o None, non {type(config)}")

        self.registry = ModelRegistry(self.config)
        self.router = Router(self.config, self.registry)
        self.metrics = MetricsLogger(self.config.metrics.db_path) if self.config.metrics.enabled else None

    def _find_config(self) -> Path:
        for path in DEFAULT_CONFIG_PATHS:
            if path.exists():
                return path
        # Nessun config trovato: mostra come crearlo
        example = Path("cheapfirst.yaml.example")
        msg = (
            "Nessun file di configurazione trovato.\n"
            f"  Crea cheapfirst.yaml nella cartella corrente copiando l'esempio:\n"
            f"    cp {example.name if example.exists() else 'cheapfirst.yaml.example'} cheapfirst.yaml\n"
            "  Oppure imposta la variabile d'ambiente DEEPSEEK_API_KEY.\n"
        )
        raise FileNotFoundError(msg)

    def decide(self, prompt: str, **kwargs) -> dict:
        """Solo decisione (dry-run): quale modello userebbe, senza eseguire."""
        messages = [{"role": "user", "content": prompt}]
        sig = classify(messages)
        decision = self.router.route(messages, sig, dry_run=True)

        if "error" in decision:
            return {
                "error": decision["error"],
                "task_type": sig.task,
                "difficulty": sig.difficulty,
                "confidence": sig.confidence,
            }

        return {
            "model": decision["model"],
            "score": decision["score"],
            "cost_est": decision["cost_est"],
            "benchmark": decision["benchmark"],
            "task_type": sig.task,
            "difficulty": sig.difficulty,
            "confidence": sig.confidence,
            "reason": decision["reason"],
            "mode": self.config.routing.mode,
            "alternatives": decision.get("alternatives", []),
        }

    def chat(self, messages: list[dict], **kwargs) -> dict:
        """Chat completa: classifica, route, esegui, verify, logga."""
        sig = classify(messages)
        result = self.router.route(messages, sig, dry_run=False)

        if self.metrics:
            log_entry = {
                "task_type": sig.task,
                "difficulty": sig.difficulty,
                "confidence": sig.confidence,
                "model_used": result.get("model", result.get("model_used", "?")),
                "turns": result.get("turns", 1),
                "cost_usd": result.get("cost_usd", 0),
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "latency_ms": result.get("latency_ms", 0),
                "success": result.get("success", True),
                "error_msg": result.get("error", None),
            }
            self.metrics.log(log_entry)

        return result

    def report(self, days: int = 7) -> str:
        """Genera report metriche."""
        if not self.metrics:
            return "Metriche disabilitate."
        return self.metrics.report(days)


__version__ = "0.1.0"
__all__ = ["CheapFirst", "TaskSignature", "classify", "load_config", "calculate_cost"]
