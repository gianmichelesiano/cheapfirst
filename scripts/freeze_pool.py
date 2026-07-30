"""Congela un pool rappresentativo di ~20 modelli da OpenRouter.

Seleziona modelli per coprire:
  - 3 ordini di grandezza di prezzo
  - Almeno un modello dominato (più caro e peggiore di un altro)
  - Un outlier di prezzo (> 30 $/M completion)
  - Uno senza benchmark (tutti None)
  - Uno con coding_index > intelligence_index
  - Uno multimodale (vision)
  - Uno con contesto piccolo (<= 16K)

Output: JSON con frozen_at, source: "openrouter", models: [...]

Uso:
    python scripts/freeze_pool.py > tests/fixtures/golden_models.json
"""

import json
from datetime import date
from urllib.request import urlopen, Request

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
USER_AGENT = "cheapfirst/0.1.0"


def _fetch_models():
    req = Request(OPENROUTER_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["data"]


def _assign_tier(completion_per_m: float, model_id: str) -> str:
    """Replica esattamente la logica di cheapfirst/registry.py::_parse_openrouter."""
    if completion_per_m <= 1.0:
        tier = "cheap"
    elif completion_per_m <= 5.0:
        tier = "mid"
    elif completion_per_m <= 20.0:
        tier = "frontier"
    else:
        tier = "ultra"
    if completion_per_m == 0 and "free" in model_id:
        tier = "free"
    return tier


def main():
    all_models = _fetch_models()
    by_id = {m["id"]: m for m in all_models}

    # Lista di modelli da mantenere (selezionati per coprire i requisiti)
    keep_ids = [
        # FREE tier — prezzo zero, "free" nell'id
        "cohere/north-mini-code:free",  # free, ha benchmark (intel=19.8, coding=36.5)
        "nvidia/nemotron-3-nano-30b-a3b:free",  # free, ha benchmark (intel=14.2)
        # CHEAP tier — 0 < comp <= 1.0
        "inclusionai/ling-2.6-flash",  # 0.03, intel=14.1 — più economico con benchmark
        "sao10k/l3-lunaris-8b",  # 0.05, benchmark=None, ctx=8192 — contesto piccolo
        "meta-llama/llama-3.1-8b-instruct",  # 0.08, intel=7.6 — DOMINATO da ling-2.6
        "ibm-granite/granite-4.1-8b",  # 0.10, intel=None, coding=9.5 — benchmark parziale null
        "qwen/qwen3.7-flash",  # 0.13, benchmark=tutti None, vision — senza benchmark + vision
        "tencent/hy3-preview",  # 0.21, intel=41.2, coding=58.8 (> intel)
        "deepseek/deepseek-v4-flash",  # 0.28, intel=40.3, coding=56.2 (> intel)
        "deepseek/deepseek-v4-pro",  # 0.87, intel=44.3, coding=59.4 (> intel)
        "nex-agi/nex-n2-pro",  # 1.00 (boundary cheap/mid), vision
        # MID tier — 1.0 < comp <= 5.0
        "minimax/minimax-m3",  # 1.20, intel=44.4, coding=58.6, vision
        "openai/gpt-5.4-nano",  # 1.25, intel=38.2, coding=56.1, vision, ctx=400K
        "qwen/qwen3.7-max",  # 4.425, intel=46, coding=66 (> intel)
        # FRONTIER tier — 5.0 < comp <= 20.0
        "x-ai/grok-4.5",  # 6.0, intel=53.8, coding=72.4, vision
        "google/gemini-3.6-flash",  # 7.5, intel=50.1, coding=69.2, vision
        "moonshotai/kimi-k3",  # 15.0, intel=57.1, coding=76.2, vision
        # ULTRA tier — comp > 20.0
        "anthropic/claude-opus-5",  # 25.0, intel=60.7, coding=78, vision
        "openai/gpt-5.6-sol",  # 30.0, intel=58.9, coding=77.4, vision
        "anthropic/claude-fable-5",  # 50.0, intel=59.9, coding=76.5 — OUTLIER di prezzo
    ]

    # Verifica che tutti i keep_ids esistano
    missing = [mid for mid in keep_ids if mid not in by_id]
    if missing:
        print(f"ERRORE: modelli non trovati su OpenRouter: {missing}", file=__import__("sys").stderr)
        __import__("sys").exit(1)

    output_models = []
    for mid in keep_ids:
        m = by_id[mid]
        provider = "openrouter"

        pricing = m.get("pricing", {})
        prompt_raw = float(pricing.get("prompt", 0))
        completion_raw = float(pricing.get("completion", 0))
        prompt_per_m = round(prompt_raw * 1_000_000, 4)
        completion_per_m = round(completion_raw * 1_000_000, 4)

        tier = _assign_tier(completion_per_m, mid)

        aa = (m.get("benchmarks") or {}).get("artificial_analysis") or {}
        benchmarks = {
            "intelligence_index": aa.get("intelligence_index", None),
            "coding_index": aa.get("coding_index", None),
            "agentic_index": aa.get("agentic_index", None),
        }

        ctx = m.get("context_length", 4096)

        caps = []
        modalities = m.get("architecture", {}).get("input_modalities") or []
        if "image" in modalities:
            caps.append("vision")

        entry = {
            "id": mid,
            "provider": provider,
            "tier": tier,
            "pricing": {
                "prompt_per_m": prompt_per_m,
                "completion_per_m": completion_per_m,
            },
            "benchmarks": benchmarks,
            "context": ctx,
            "caps": caps,
        }
        output_models.append(entry)

    output = {
        "frozen_at": date.today().isoformat(),
        "source": "openrouter",
        "models": output_models,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
