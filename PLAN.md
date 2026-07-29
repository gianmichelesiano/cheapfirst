# cheapfirst — Piano di Implementazione

## Stato attuale: ✅ COMPLETATO (tranne PyPI publish)

Repo: https://github.com/gianmichelesiano/cheapfirst

## Fasi

### ✅ Fase 1: Registry + Classifier
- Classificatore euristico (regex): code, math, creative, factual, translation, general
- Registry da OpenRouter API (fetch + parse + cache, 367+ modelli)
- Modelli custom da config YAML
- Filtro per API key attive

### ✅ Fase 2: Router + Executor reali
- Router: filtro per competenza + ranking costo/benchmark
- Executor: chiamate API HTTP reali OpenAI-compatibili
- 12 provider mappati (DeepSeek, Anthropic, OpenAI, Groq, ecc.)
- Provider sconosciuti: inferiti dal nome
- Calcolo costo effettivo da token reali
- Timeout e error handling

### ✅ Fase 3: Verify + Escalate
- Verify strutturato per tipo di task (codice, traduzione, matematica)
- Controllo parentesi bilanciate per codice
- Lunghezza minima contestuale
- Escalation ladder con max_turns
- Budget controllo costi per verify

### ✅ Fase 4: Metriche + Report
- Logging richieste su SQLite
- Report testuale: costi, modelli, success rate
- Query per periodo

### ✅ Fase 5: CLI completa
- `cheapfirst route` — routing + esecuzione
- `cheapfirst decide` — dry-run
- `cheapfirst registry update/check`
- `cheapfirst report --days N`
- `cheapfirst serve`

### ✅ Fase 6: Server HTTP (extra [server])
- FastAPI endpoint
- POST /v1/chat/completions (OpenAI-compatibile)
- GET /v1/models
- GET /v1/route (dry-run)
- GET /healthz
- Passthrough per modelli specifici

### ✅ Fase 7: Testing
- 27 test passanti:
  - 12 test classifier
  - 5 test registry
  - 3 test router
  - 7 test verify

### ⬜ Fase 8a: pip publish
- Build wheel + tar.gz: ✅ fatto (dist/)
- PyPI publish: ❌ serve PyPI API token
  - Crea account su pypi.org
  - Genera un token API
  - `python -m twine upload dist/*`

### ⬜ Fase 8b: Articolo tecnico
- Da scrivere

## Struttura del progetto

```
cheapfirst/
├── cheapfirst/
│   ├── __init__.py         # CheapFirst class pubblica
│   ├── __version__.py      # v0.1.0
│   ├── __main__.py         # Entry point
│   ├── config.py           # Lettura YAML + mappa provider
│   ├── classifier.py       # Classificatore euristico
│   ├── router.py           # Motore di routing
│   ├── registry.py         # Registry modelli (OpenRouter + custom)
│   ├── executor.py         # Chiamata API reale
│   ├── verify.py           # Verifica risposta
│   ├── metrics.py          # Metriche SQLite
│   ├── report.py           # Report metriche
│   ├── server.py           # Server HTTP FastAPI
│   └── cli/
│       └── main.py         # CLI (route, decide, registry, report, serve)
├── tests/
│   ├── test_classifier.py  # 12 test
│   ├── test_registry.py    # 5 test
│   ├── test_router.py      # 3 test
│   └── test_verify.py      # 7 test
├── pyproject.toml
├── README.md
├── SPECS.md
├── PLAN.md
└── dist/                   # build pronto per PyPI
```

## Totale: ~2100 righe di codice, 27 test passanti
