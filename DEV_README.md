# Local Development

## Configure Test PyPI for Poetry

For publishing packages to Test PyPI using Poetry during development, it is recommended to use an environment variable for the Test PyPI token. This avoids issues with keychain access on macOS and is generally simpler for automation and scripting.

### Setting Up Environment Variable for Test PyPI Token

1. **Export the Test PyPI API Token**:

Export your Test PyPI token as an environment variable in your terminal session before running the publish command.

```bash
export POETRY_TEST_PYPI_API_TOKEN=<your_test_pypi_token>
```

2. **Run the local publish command**:

In the project root folder.

```bash
./tests/build_test_pypi.sh
```