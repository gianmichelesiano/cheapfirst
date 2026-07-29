# cheapfirst

LLM router: prova il cheap, verifica, scala. Risparmia fino all'80% sui costi API.

## Come funziona

```
richiesta → classifica (euristico, 0ms)
         → filtra modelli competenti per il task
         → ranka per costo/benchmark
         → esegue il miglior rapporto qualità/prezzo
         → se necessario, verifica e scala al tier successivo
```

## Installazione

```bash
pip install cheapfirst
```

## Quickstart

```bash
# Crea un file di configurazione
cp cheapfirst.yaml.example cheapfirst.yaml
# Inserisci le tue API key (o usa variabili d'ambiente)

# Provalo
cheapfirst route "Traduci hello in italiano"
cheapfirst decide "Spiegami la relatività generale"
```

Oppure in Python:

```python
from cheapfirst import CheapFirst

router = CheapFirst()

# Chat completa
response = router.chat([
    {"role": "user", "content": "Ciao, traduci 'hello' in italiano"}
])
print(response["text"])          # "Ciao"
print(response["model_used"])    # "deepseek/deepseek-v4-flash"
print(response["cost_usd"])      # 0.0000096

# Solo decisione (dry-run, senza eseguire)
decision = router.decide("Progetta un sistema di rate limiting")
print(decision["model"])         # "deepseek/deepseek-v4-pro"
print(decision["score"])         # 0.0146
```

## Comandi CLI

```bash
cheapfirst route "prompt"              # Routing + esecuzione
cheapfirst decide "prompt"             # Solo decisione (dry-run)
cheapfirst registry update             # Aggiorna modelli da OpenRouter
cheapfirst report --days 7             # Report metriche
cheapfirst serve --port 8080           # Server HTTP (extra [server])
```

## Configurazione

Vedi `cheapfirst.yaml.example`. I provider senza API key vengono ignorati automaticamente.

## Licenza

MIT
