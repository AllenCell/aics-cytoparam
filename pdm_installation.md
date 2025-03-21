# Installation and usage with pdm

1. [Install pdm](https://pdm-project.org/en/latest/#recommended-installation-method)
2. Install dependencies: `pdm sync --no-isolation`.
3. Prefix every `python` command with `pdm run`. For example:

```
pdm run python src/aicscytoparam/tests/dummy_test.py
```