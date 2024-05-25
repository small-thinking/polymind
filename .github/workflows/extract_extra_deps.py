# .github/workflows/extract_extra_deps.py
import toml

pyproject = toml.load("pyproject.toml")
extras = " ".join([f"-E {key}" for key in pyproject["tool"]["poetry"]["extras"].keys()])
print(extras)
