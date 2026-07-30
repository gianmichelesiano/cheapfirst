# cheapfirst

**LLM router: try cheap, verify, escalate.**

```
request ──→ classify (heuristic, 0ms, $0)
         ──→ filter by minimum quality (difficulty-based threshold)
         ──→ pick cheapest among qualified models
         ──→ execute, verify response, escalate to stronger model if needed
```

## Installation

```bash
pip install cheapfirst
pip install cheapfirst[server]  # optional HTTP server
```

## Quickstart

```bash
# 1. Get an OpenRouter API key: https://openrouter.ai/keys
# 2. Create config
cp cheapfirst.yaml.example cheapfirst.yaml
# Set OPENROUTER_API_KEY in your environment or edit the YAML
# 3. Update model registry
cheapfirst registry update
# 4. Test
cheapfirst route "Translate hello to Italian"
```

## Examples

### Python

```python
from cheapfirst import CheapFirst

router = CheapFirst()

response = router.chat([
    {"role": "user", "content": "Explain general relativity"}
])
print(response["text"])
print(response["model_used"])
print(response["cost_usd"])

# Dry-run (classify + rank, no API call)
decision = router.decide("Design a distributed rate limiter")
print(decision["model"])
print(decision["score"])
print(decision["alternatives"])
```

### CLI

```bash
cheapfirst route "What is the capital of France?"
cheapfirst decide "How does a transformer work?"  # dry-run
cheapfirst registry update
cheapfirst report --days 7
cheapfirst serve --port 8080
```

### HTTP Server (OpenAI-compatible)

```bash
cheapfirst serve --port 8080
```

Then use any OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="cheapfirst-auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## How it decides

**1. Classify the task** — heuristic regex rules, 0ms, free

**2. Quality filter** — difficulty-based minimum benchmark threshold:
   - Easy (difficulty < 0.30): bench >= 25
   - Medium (0.30-0.70): bench >= 35-50
   - Hard (> 0.70): bench >= 50-65

**3. Pick cheapest** — among qualified models, best cost/benchmark ratio

**4. High confidence (>80%)** — execute directly, one call

**5. Verify & escalate** — check response quality, escalate to stronger models if needed

## Data sources

| Data | Source | Coverage |
|------|--------|----------|
| Model pricing | OpenRouter API | 367 models |
| Benchmarks | Artificial Analysis via OpenRouter | 136 models |
| Custom/local models | YAML config | user-defined |

> Only models with Artificial Analysis benchmarks are routed. 136 models from major providers.

## Configuration

```yaml
# cheapfirst.yaml
provider_keys:
  openrouter: ${OPENROUTER_API_KEY}

routing:
  verify: true
  max_turns: 3
  skip_verify_confidence: 0.8
```

## Architecture

```
cheapfirst/
├── cheapfirst/
│   ├── classifier.py     # Heuristic task classifier (regex, 0ms)
│   ├── router.py         # Quality filter + cost ranking + escalate
│   ├── registry.py       # Model registry (OpenRouter + custom)
│   ├── executor.py       # API caller (OpenRouter-native)
│   ├── verify.py         # Response verification
│   ├── metrics.py        # SQLite logging + reports
│   ├── server.py         # FastAPI HTTP server
│   └── cli/main.py       # CLI
├── tests/                # 27 tests
├── pyproject.toml
└── README.md
```

## License

MIT
