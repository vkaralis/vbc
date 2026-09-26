# Contributing

Contributions are welcome under the repository's MIT License.

1. Create a focused branch.
2. Install the development dependencies with `python -m pip install -e '.[dev]'`.
3. Add tests for numerical or behavioural changes.
4. Run `pytest` and `ruff check .`.
5. Explain scientific assumptions and cite relevant sources in the pull request.

Changes affecting clinical or regulatory interpretation should include independent numerical
validation against a trusted implementation or reference dataset.
