Vedi `README.md` per quickstart.

## Struttura del progetto

```
cheapfirst/
├── cheapfirst/                  # Pacchetto principale
│   ├── __init__.py              # CheapFirst class
│   ├── __version__.py           # Versione
│   ├── __main__.py              # CLI entry point
│   ├── config.py                # Lettura YAML config
│   ├── classifier.py            # Classificatore euristico
│   ├── router.py                # Motore di routing
│   ├── registry.py              # Registry modelli (OpenRouter + custom)
│   ├── executor.py              # Chiamata API
│   ├── verify.py                # Verifica risposta
│   ├── metrics.py               # Metriche SQLite
│   ├── report.py                # Report metriche
│   └── cli/
│       ├── __init__.py
│       └── main.py              # CLI (route, decide, registry, report, serve)
├── tests/
│   └── test_classifier.py
├── cheapfirst.yaml.example
├── pyproject.toml
├── README.md
├── LICENSE
└── SPECS.md
```

## Fonti dati

- **Prezzi + benchmark**: OpenRouter API `/api/v1/models` (367+ modelli)
- **Modelli custom**: YAML config (modelli locali gratuiti)

## Benchmark mappa

| Task type | Benchmark |
|-----------|-----------|
| `code` | `coding_index` (Artificial Analysis) |
| `math` | `intelligence_index` |
| `creative` | `intelligence_index` |
| `translation` | `intelligence_index` |
| `general` | `intelligence_index` |

## Competitor

- **Maestro** — TypeScript, v0.1, solo OpenRouter, 5 ore di build
- **RouteWise** (Harvard) — solo matematica costi, non classifica task
- **Provara** — BSL license, TypeScript, SaaS-oriented
- **LiteLLM** — proxy generico, non classifica
