import sys

packages = [
    "qdrant_client",
    "fastembed",
    "datasets",
    "sentence_transformers",
    "tqdm",
    "google.generativeai"
]

print("Python version:", sys.version)
print("\nChecking packages:")
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  {pkg}: Installed")
    except ImportError:
        print(f"  {pkg}: MISSING")
