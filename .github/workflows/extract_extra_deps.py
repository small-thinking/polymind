# .github/workflows/extract_extras.py
import toml

pyproject = toml.load("pyproject.toml")
extras = ",".join(pyproject["tool"]["poetry"]["extras"].keys())
print(extras)
