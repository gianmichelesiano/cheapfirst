"""Adapter universale OpenAI-compatibile per chiamate API reali."""

import time
import json
import httpx
from .config import PROVIDER_BASE_URLS


def get_provider_info(
    model_id: str,
    provider_keys: dict,
    provider: str | None = None,
) -> tuple[str | None, str | None]:
    """Restituisce (base_url, api_key) per un dato model_id.

    Il model_id è nel formato "provider/modello" (es. "deepseek/deepseek-v4-flash").
    Il provider risolto è:
      - Il parametro ``provider`` (passato dal ModelSpec, fonte di verità)
      - In assenza, model_id.split("/")[0] (fallback per chiamate dirette)

    Se il provider è nella mappa PROVIDER_BASE_URLS, usa quella URL + API key.
    Se non è nella mappa ma la provider_keys ha un URL come valore
    (es. "local": "http://localhost:11434/v1"), usa quello (locale/proxy).
    Altrimenti solleva ValueError.
    """
    # Fonte di verità: se abbiamo il provider dal ModelSpec, usiamo quello
    resolved_provider = provider if provider else model_id.split("/")[0]

    # Mappa predefinita
    if resolved_provider in PROVIDER_BASE_URLS:
        base_url = PROVIDER_BASE_URLS[resolved_provider]
        api_key = provider_keys.get(resolved_provider, "")
        return base_url, api_key

    # Provider non standard: controlla se la key è un URL (locale/proxy)
    provider_val = provider_keys.get(resolved_provider, "")
    if provider_val and provider_val.startswith("http"):
        base_url = provider_val.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        return base_url, "not-needed"

    # Nessuna informazione utile — fallisci esplicitamente
    raise ValueError(
        f"Provider sconosciuto: '{resolved_provider}' (da model_id='{model_id}'). "
        f"Aggiungi '{resolved_provider}' a provider_keys nella configurazione "
        f"o a PROVIDER_BASE_URLS nel codice."
    )


def calculate_cost(model_pricing: dict, input_tokens: int, output_tokens: int) -> float:
    """Calcola il costo in USD dati i prezzi per milione di token."""
    cost = 0.0
    cost += (input_tokens * model_pricing.get("prompt_per_m", 0)) / 1_000_000
    cost += (output_tokens * model_pricing.get("completion_per_m", 0)) / 1_000_000
    return cost


def execute(
    messages: list[dict],
    model_id: str,
    provider_keys: dict = None,
    provider: str | None = None,
    **kwargs,
) -> dict:
    """Esegue una chiamata API reale a un modello OpenAI-compatibile.

    Args:
        messages: Lista di messaggi nel formato OpenAI.
        model_id: ID del modello (es. "deepseek/deepseek-v4-flash").
        provider_keys: dict con {provider: api_key}.
        provider: Nome del provider (fonte di verità dal ModelSpec).
                  Se None, estratto da model_id.split("/")[0].
        **kwargs: temperature, max_tokens, stream, ecc.

    Returns:
        dict con: text, model, input_tokens, output_tokens, cost_usd, latency_ms
    """
    if provider_keys is None:
        provider_keys = {}

    try:
        base_url, api_key = get_provider_info(
            model_id, provider_keys, provider=provider,
        )
    except ValueError as e:
        return {
            "error": f"Provider non configurato per {model_id}. {e}",
            "success": False,
        }

    # Costruisci URL: assicurati che finisca con /chat/completions
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model_id.split("/", 1)[1] if "/" in model_id else model_id,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 4096),
    }

    if kwargs.get("stream"):
        body["stream"] = True

    start = time.time()

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = int((time.time() - start) * 1000)

        # Estrai risposta dal formato OpenAI
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "")

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return {
            "text": text,
            "model": model_id,
            "model_used": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": 0,  # calcolato dopo dal router
            "latency_ms": latency_ms,
            "success": True,
            "provider": model_id.split("/")[0],
        }

    except httpx.HTTPStatusError as e:
        latency_ms = int((time.time() - start) * 1000)
        error_detail = ""
        try:
            error_detail = e.response.text[:300]
        except Exception:
            error_detail = str(e)
        return {
            "error": f"HTTP {e.response.status_code}: {error_detail}",
            "model": model_id,
            "success": False,
            "latency_ms": latency_ms,
        }
    except httpx.TimeoutException:
        latency_ms = int((time.time() - start) * 1000)
        return {
            "error": f"Timeout dopo {latency_ms}ms chiamando {model_id}",
            "model": model_id,
            "success": False,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        return {
            "error": str(e),
            "model": model_id,
            "success": False,
            "latency_ms": latency_ms,
        }
