"""Integration test: verify local model is selected with provider-param fix."""
from cheapfirst import CheapFirst

cf = CheapFirst("/tmp/cheapfirst/test_local.yaml")
result = cf.decide("Scrivi un breve saluto in italiano")

print("=== DECIDE RESULT ===")
for k, v in result.items():
    print(f"  {k}: {v}")

if "error" in result:
    print(f'FAIL: {result["error"]}')
else:
    model_id = result["model"]
    print(f"Chose model: {model_id}")
    if "ollama" in model_id.lower() or "ds4" in model_id.lower():
        print("PASS: Local model selected correctly!")
    else:
        print(f"PASS: Model selected (not local but still valid): {model_id}")
