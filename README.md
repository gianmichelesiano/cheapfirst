# cheapfirst

**LLM router: try cheap, verify, escalate. Save up to 80% on API costs without sacrificing quality.**

```
request ──→ classify (heuristic, 0ms, $0)
         ──→ filter competent models for the task
         ──→ rank by cost/benchmark (lowest score = best quality per dollar)
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
# 2. Add your API keys (DeepSeek, Anthropic, OpenAI...)
# 3. Test the router
cheapfirst route "Translate hello to Italian"
# Output: Ciao | Model: deepseek/deepseek-v4-flash | Cost: $0.0000096
```

## Examples

### Python

```python
from cheapfirst import CheapFirst

router = CheapFirst()

# Full routing + execution
response = router.chat([
    {"role": "user", "content": "Explain general relativity"}
])
print(response["text"])          # the response
print(response["model_used"])    # "deepseek/deepseek-v4-pro"
print(response["cost_usd"])      # actual cost in $

# Dry-run (classify + rank, no API call)
decision = router.decide("Design a distributed rate limiter")
print(decision["model"])         # recommended model
print(decision["score"])         # 0.0146 (cost/benchmark)
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

**2. Filter competent models** — removes models with low benchmark scores for that task type

**3. Rank by cost ÷ benchmark** — simple math:
   ```
   score = (output_price × estimated_tokens) / benchmark_score
   Lower score = better quality per dollar
   ```

**4. High confidence (>80%)** → execute directly (one call, zero overhead)

**5. Low confidence** → try cheap model, verify response, escalate if needed (max 3 turns)

## Configuration

```yaml
# cheapfirst.yaml
provider_keys:
  deepseek: ${DEEPSEEK_API_KEY}
  anthropic: ${ANTHROPIC_API_KEY}
  openai: ${OPENAI_API_KEY}

routing:
  verify: true                    # enable verify/escalate
  max_turns: 3                    # max escalation turns
  skip_verify_confidence: 0.8     # skip verify if confidence > 80%
```

Providers without API keys are ignored automatically. Add as many as you want.

## Data sources

| Data | Source |
|------|--------|
| Model pricing | OpenRouter API (367+ models, auto-updated) |
| Benchmarks | Artificial Analysis (intelligence, coding, agentic index) |
| Custom/local models | YAML config (zero cost, manual benchmarks) |

## Supported providers

Any OpenAI-compatible API: DeepSeek, Anthropic, OpenAI, Groq, Google Gemini, Mistral, Together AI, Fireworks, Cohere, Perplexity, xAI, GitHub Models + any local endpoint (Ollama, vLLM, llama.cpp, ds4).

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
# 27 passing
```

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
├── tests/                # 27 tests
├── pyproject.toml
└── README.md
```

## License

MIT
