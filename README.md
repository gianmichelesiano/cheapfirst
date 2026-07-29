# cheapfirst

LLM router: prova il modello cheap, verifica, scala. Risparmia fino all'80% sui costi API.

```
richiesta ──→ classifica (euristico, 0ms, $0)
           ──→ filtra modelli competenti per il task
           ──→ ranka per costo/benchmark (costo ÷ qualità)
           ──→ esegue il miglior rapporto qualità/prezzo
           ──→ verifica la risposta, scala se necessario
```

## Installazione

```bash
pip install cheapfirst

# Per il server HTTP
pip install cheapfirst[server]
```

## Quickstart

```bash
# 1. Crea config
cp cheapfirst.yaml.example cheapfirst.yaml
# 2. Inserisci le tue API key (DeepSeek, Anthropic, OpenAI...)
# 3. Prova il routing
cheapfirst route "Traduci hello in italiano"
# Output: Ciao | Modello: deepseek/deepseek-v4-flash | Costo: $0.0000096
```

## Esempi

### Python

```python
from cheapfirst import CheapFirst

router = CheapFirst()

# Routing completo + esecuzione
response = router.chat([
    {"role": "user", "content": "Spiegami la relatività generale"}
])
print(response["text"])          # spiegazione
print(response["model_used"])    # "deepseek/deepseek-v4-pro"
print(response["cost_usd"])      # costo effettivo in $

# Solo decisione (dry-run, 0 costi)
decision = router.decide("Progetta un sistema di rate limiting")
print(decision["model"])         # modello raccomandato
print(decision["score"])         # 0.0146 (costo/benchmark)
print(decision["alternatives"])  # altre opzioni con punteggio
```

### CLI

```bash
# Routing + esecuzione
cheapfirst route "Traduci ciao in inglese"

# Solo decisione (nessuna chiamata API)
cheapfirst decide "Come funziona un transformer?"

# Gestione registry modelli
cheapfirst registry update     # Scarica 367+ modelli da OpenRouter
cheapfirst registry check      # Stato del registry

# Report metriche
cheapfirst report --days 7

# Server HTTP (OpenAI-compatibile)
cheapfirst serve --port 8080
```

### Server HTTP

```bash
pip install cheapfirst[server]
cheapfirst serve --port 8080
```

Poi usa qualsiasi client OpenAI:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

# Routing automatico
resp = client.chat.completions.create(
    model="costflow-auto",
    messages=[{"role": "user", "content": "Ciao!"}],
)
print(resp.choices[0].message.content)
```

## Configurazione

`cheapfirst.yaml`:

```yaml
provider_keys:
  deepseek: ${DEEPSEEK_API_KEY}
  anthropic: ${ANTHROPIC_API_KEY}
  openai: ${OPENAI_API_KEY}

routing:
  verify: true                    # abilita verify/escalate
  max_turns: 3                    # max escalation
  skip_verify_confidence: 0.8     # skip verify se confidenza > 80%
```

I provider senza API key vengono ignorati automaticamente. Puoi avere quanti ne vuoi.

## Come decide cheapfirst?

**1. Classifica il task** — regex euristiche, 0ms, $0:
   - `"traduci ..."` → translation
   - `"``` ... def ..."` → code  
   - `"progetta ... complesso"` → complesso (difficoltà > 0.7)
   
**2. Filtra modelli competenti** — esclude chi ha benchmark basso per quel task

**3. Ranka per costo ÷ benchmark** — formula matematica:
   ```
   score = (prezzo_output × token_stimati) / benchmark_score
   Più basso = meglio = più qualità per dollar
   ```

**4. Se confidenza alta (>80%)** → esegue direttamente (una chiamata, zero overhead)

**5. Se confidenza bassa** → prova modello cheap, verifica risposta, scala se serve (max 3 turni)

## Fonti dati

| Dato | Fonte |
|------|-------|
| Prezzi modelli | OpenRouter API (367+ modelli, aggiornato) |
| Benchmark | Artificial Analysis (intelligence, coding, agentic index) |
| Modelli locali | Config YAML (costo zero, benchmark stimati) |

## Provider supportati

Tutti i provider con API OpenAI-compatibile: DeepSeek, Anthropic, OpenAI, Groq, Google Gemini, Mistral, Together, Fireworks, Cohere, Perplexity, xAI, GitHub Models + qualsiasi locale (Ollama, vLLM, llama.cpp, ds4).

## Test

```bash
pip install -e . pytest
python -m pytest tests/ -v
```

## Progetti simili

| Progetto | Differenza |
|----------|-----------|
| **Maestro** | TypeScript, v0.1, solo OpenRouter |
| **RouteWise** (Harvard) | Solo matematica costi, non classifica task |
| **LiteLLM** | Proxy generico, non ha routing intelligente |
| **Provara** | BSL license, SaaS-oriented |

## Licenza

MIT
