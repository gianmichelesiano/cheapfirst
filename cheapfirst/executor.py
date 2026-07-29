"""Adapter universale OpenAI-compatibile per chiamate API."""

import time
from openai import OpenAI


def execute(messages: list[dict], model_id: str, **kwargs) -> dict:
    """Esegue una chiamata a un modello OpenAI-compatibile.

    Args:
        messages: Lista di messaggi nel formato OpenAI.
        model_id: ID del modello (es. "deepseek/deepseek-v4-flash").
        **kwargs: Parametri extra (temperature, max_tokens, etc.).

    Returns:
        dict con: text, model, input_tokens, output_tokens, cost_usd, latency_ms
    """
    provider = model_id.split("/")[0]

    start = time.time()

    try:
        # Nota: in v0.1 usiamo una chiamata HTTP diretta
        # In futuro: client configurato con base_url e api_key per provider
        import httpx

        # Determina base_url e api_key dal provider
        # Per ora: chiamata generica
        response = _call_openai_compatible(messages, model_id)

        latency_ms = int((time.time() - start) * 1000)
        text = response.get("text", "")
        input_tokens = response.get("input_tokens", 0)
        output_tokens = response.get("output_tokens", 0)

        return {
            "text": text,
            "model": model_id,
            "model_used": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": response.get("cost", 0),
            "latency_ms": latency_ms,
            "success": True,
        }

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        return {
            "error": str(e),
            "model": model_id,
            "success": False,
            "latency_ms": latency_ms,
        }


def _call_openai_compatible(messages: list[dict], model_id: str) -> dict:
    """Chiamata HTTP diretta a un endpoint OpenAI-compatibile.

    Nota: implementazione base. In v0.2 useremo il client OpenAI
    configurato per provider.
    """
    # Placeholder: in v0.1 restituisce un risultato mock
    # La vera implementazione richiede di risolvere base_url e api_key
    # per il provider specifico dal model_id

    # Per test: restituisce un risultato finto
    return {
        "text": f"[cheapfirst] Risposta da {model_id} (placeholder)",
        "input_tokens": 50,
        "output_tokens": 100,
        "cost": 0.0001,
    }
