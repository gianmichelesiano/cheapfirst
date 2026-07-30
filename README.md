# cheapfirst

**LLM router: try cheap, verify, escalate. Save up to 80% on API costs without sacrificing quality.**

```
request ──→ classify (heuristic, 0ms, $0)
         ──→ filter competent models for the task
         ──→ rank by quality floor + cheapest
         ──→ execute the best model
         ──→ verify response, escalate if needed (max 3 turns)
```

## Installation

```bash
pip install cheapfirst

# For the HTTP server (optional)
pip install cheapfirst[server]
```

## Quickstart

```bash
# 1. Create config
cp cheapfirst.yaml.example cheapfirst.yaml
# 2. Add your API keys (at least one)
# 3. Test the router
cheapfirst route "Translate hello to Italian"
# Output: Ciao | Model: deepseek/deepseek-v4-flash | Cost: $0.0000096
```

## Features

- **Classifier ibrido** — euristico (0ms, $0) per task ovvii, LLM opzionale per casi ambigui
- **OpenRouter come fonte di verità** — 367+ modelli con prezzi e benchmark Artificial Analysis
- **OpenRouter come executor** — una sola API key per tutti i modelli del catalogo
- **Ranking quality/price** — `quality_floor(difficoltà) + cheapest` con lambda tie-break su benchmark
- **Verify/escalate** — a cascata (max 3 turni) se confidenza bassa
- **Provider universale** — qualsiasi API OpenAI-compatibile (locale, cloud, proxy)
- **Metriche SQLite** — report settimanali con costi reali

## Quick example

```python
from cheapfirst import CheapFirst

router = CheapFirst()

# Full routing + execution
response = router.chat([
    {"role": "user", "content": "Explain general relativity"}
])
print(response["text"])          # the response
print(response["model_used"])    # e.g. "deepseek/deepseek-v4-pro"
print(response["cost_usd"])      # actual cost in $

# Dry-run (classify + rank, no API call)
decision = router.decide("Design a distributed rate limiter")
print(decision["model"])         # recommended model
print(decision["score"])         # ranking score
print(decision["alternatives"])  # other options with scores
```

### CLI

```bash
# Route + execute
cheapfirst route "What is the capital of France?"

# Dry-run (no API call)
cheapfirst decide "How does a transformer work?"

# Update model registry from OpenRouter
cheapfirst registry update

# Show report
cheapfirst report --days 7

# Start HTTP server (requires [server] extra)
cheapfirst serve --port 8080
```

### HTTP Server

```bash
pip install cheapfirst[server]
cheapfirst serve --port 8080
```

Then use any OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

response = client.chat.completions.create(
    model="costflow-auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## How it decides

**1. Classify the task** — heuristic rules (regex), 0ms, free:
   - `"translate hello to Italian"` → translation
   - `"write a Python function..."` → code
   - `"design a complex distributed system"` → high difficulty
   - Difficulty: 0.1 (translation) → 1.0 (complex system design)

**2. Filter competent models** — removes models with benchmark scores below the task's minimum threshold. Models without benchmarks are handled via `unmeasured_policy` (default: `exclude`, alternative: `impute_from_tier`).

**3. Rank by quality floor + cheapest**:
   ```
   floor = linear_map(difficulty)         # 0.2→30, 0.5→45, 0.8→58, 1.0→65
   qualified = models where benchmark >= floor
   ranked = qualified sorted by cost ascending
   tie-break: if cost diff < 5%, higher benchmark wins
   ```

**4. High confidence (>80%)** → execute directly (one call, zero overhead)

**5. Low confidence** → try cheapest qualified model, verify response, escalate if needed (max 3 turns to progressively better models)

## Model registry

cheapfirst scarica automaticamente il catalogo modelli da OpenRouter API (`/api/v1/models`) con prezzi e benchmark Artificial Analysis. Una volta popolato:

- Tutti i modelli del catalogo sono eseguibili via **OpenRouter** con una sola API key
- I provider nativi (DeepSeek, OpenAI, Anthropic, ecc.) funzionano direttamente per i loro modelli
- I provider locali (Ollama, vLLM, llama.cpp, ds4) vanno configurati in `provider_keys` con URL

## Configuration

```yaml
# cheapfirst.yaml
provider_keys:
  openrouter: ${OPENROUTER_API_KEY}   # unlocks all OpenRouter models
  deepseek: ${DEEPSEEK_API_KEY}       # direct access (bypass OpenRouter)
  anthropic: ${ANTHROPIC_API_KEY}
  openai: ${OPENAI_API_KEY}
  local: "http://localhost:11434/v1"  # local providers use URL as value

routing:
  verify: true                    # enable verify/escalate
  max_turns: 3                    # max escalation turns
  skip_verify_confidence: 0.8    # skip verify if confidence > 80%
  unmeasured_policy: exclude      # exclude | impute_from_tier (default: exclude)
```

Providers without API keys are ignored automatically. Add as many as you want.

## Data sources

| Data | Source |
|------|--------|
| Model pricing | OpenRouter API (367+ models, auto-updated) |
| Benchmarks | Artificial Analysis (intelligence, coding, agentic index) |
| Custom/local models | YAML config (zero cost, manual benchmarks) |

## Supported providers

### OpenRouter (recommended)
Una chiave API per tutti i modelli del catalogo: DeepSeek, Anthropic, OpenAI, Google Gemini, Groq, Mistral, Together AI, Fireworks, Cohere, Perplexity, xAI, GitHub Models e centinaia di altri. Il model ID completo (es. `deepseek/deepseek-v4-flash`) viene passato direttamente all'API.

### Provider nativi
Qualsiasi API OpenAI-compatibile configurata in `provider_keys` con `base_url`:

- DeepSeek, Anthropic, OpenAI — mappati automaticamente in `PROVIDER_BASE_URLS`
- Locali: Ollama (`http://localhost:11434/v1`), vLLM, llama.cpp, ds4 — basta aggiungere l'URL in `provider_keys`

## How it compares

| Project | Why cheapfirst is different |
|---------|---------------------------|
| **Maestro** | TypeScript, v0.1, 5-hour build, OpenRouter-only |
| **RouteWise** (Harvard) | Cost math only, no task classification, academic paper |
| **LiteLLM** | Generic proxy, no intelligent routing |
| **RouteLLM** | Only 2 models (weak/strong), not multi-provider |
| **Provara** | BSL license, TypeScript, SaaS-oriented |

## Tests

```bash
pip install -e . pytest
python -m pytest tests/ -v
# 39 non-spec tests passing
```

Lab tests (OpenRouter API) sono marcati `@pytest.mark.spec` e esclusi dal default.

## Architecture

```
cheapfirst/
├── cheapfirst/
│   ├── classifier.py     # Heuristic task classifier (regex, 0ms)
│   ├── router.py         # Filter + rank + decide + escalate
│   ├── registry.py       # Model registry (OpenRouter + custom)
│   ├── executor.py       # Universal OpenAI-compatible API caller
│   ├── verify.py         # Response verification (code, translation, math)
│   ├── metrics.py        # SQLite logging + reports
│   ├── server.py         # FastAPI HTTP server
│   └── cli/main.py       # CLI (route, decide, registry, report, serve)
├── scripts/
│   ├── freeze_pool.py    # Congela snapshot rappresentativo da OpenRouter
│   └── ...                # Utility scripts
├── tests/                # 39 non-spec tests
├── pyproject.toml
└── README.md
```

## License

MIT
