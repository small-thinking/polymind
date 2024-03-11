# Local Development

## Configure Test Pypi
1. Add the Test PyPI Repository to Poetry:

You only need to do this once per machine. This step tells Poetry where the Test PyPI repository is located.
```bash
poetry config repositories.testpypi https://test.pypi.org/legacy/
```

2. Set the Test PyPI API Token:

Replace <your_test_pypi_token> with your actual Test PyPI token. This token is used for authentication when publishing packages to Test PyPI.

```bash
poetry config http-basic.testpypi __token__ <your_test_pypi_token>
```