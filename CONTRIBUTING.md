# Contributing

Thank you for your interest in contributing to the 6-DOF Launch Vehicle Ascent Simulation!

## Getting Started

1. Fork the repository and create a branch from `main`.
2. Install the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Making Changes

- Keep changes focused — one logical change per pull request.
- Follow the existing code style. The pre-commit hooks will run automatically, or run manually:
  ```bash
  ruff check .
  ruff format .
  ```
- Add or update tests in `tests/` for any new functionality.
- Ensure all tests pass:
  ```bash
  pytest tests/ -v
  ```

## Pull Request Process

1. Fill in the pull request template.
2. Ensure CI passes (lint + tests on Python 3.11, 3.12, and 3.13).
3. Describe what you changed and why in the PR description.

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests.

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## License

By contributing you agree that your contributions will be licensed under the [MIT License](LICENSE).
