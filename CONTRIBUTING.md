# Contributing to cylinderfit2026

Thank you for your interest in improving cylinderfit2026! This document walks you
through everything you need to go from idea to merged pull request.

---

## Getting started

### 1. Fork and clone

```bash
# Fork on GitHub first, then:
git clone https://github.com/<your-username>/cylinderfit2026.git
cd cylinderfit2026
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install in editable mode with dev dependencies

```bash
pip install -e ".[dev]"
```

The `[dev]` extra pulls in pytest, hypothesis, ruff, mypy, sphinx, and all other
tools needed for development.

---

## Running tests

Run the full test suite (unit, property-based, regression, and bias tests):

```bash
python -m pytest tests/ -q
```

Run a specific test file:

```bash
python -m pytest tests/test_ransac.py -q
```

Run with coverage:

```bash
python -m pytest tests/ --cov=cylinderfit2026 --cov-report=term-missing
```

The CI matrix runs Python 3.9–3.12 on Ubuntu, macOS, and Windows. Please make
sure your changes do not break any combination before opening a PR.

---

## Linting and type-checking

We use **ruff** for linting and formatting, and **mypy** for static type checking.

```bash
# Lint source and tests
ruff check src/ tests/

# Auto-fix safe issues
ruff check --fix src/ tests/

# Check formatting
ruff format --check src/ tests/

# Apply formatting
ruff format src/ tests/

# Type-check
mypy src/cylinderfit2026
```

All of the above must pass cleanly before a PR can be merged. The CI workflow
enforces this automatically.

---

## Submitting a pull request

### Branch naming

Use a short, descriptive kebab-case name prefixed with the change type:

| Prefix | Use for |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `refactor/` | Code restructuring without behaviour change |
| `docs/` | Documentation only |
| `test/` | Adding or improving tests |
| `ci/` | CI/CD changes |

Examples: `feat/gpu-ransac`, `fix/cone-jacobian-singularity`, `docs/open3d-adapter`

### Commit style

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short imperative summary>

[optional body — explain the *why*, not the *what*]

[optional footer(s), e.g. Closes #42]
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, `chore`, `perf`

### PR checklist

Before requesting a review, confirm:

- [ ] All tests pass locally (`python -m pytest tests/ -q`)
- [ ] `ruff check src/ tests/` reports zero issues
- [ ] `ruff format --check src/ tests/` reports zero issues
- [ ] `mypy src/cylinderfit2026` exits cleanly
- [ ] Golden-value pins are updated if any fitting algorithm changed (see below)
- [ ] New public API surfaces are documented with NumPy-style docstrings
- [ ] CHANGELOG.md has an entry under `[Unreleased]`
- [ ] The PR description explains *what* changed and *why*

---

## Updating golden-value pins

Golden-value tests (`tests/test_golden.py`) store expected parameter arrays that
are produced by fitting specific, deterministic point clouds. They exist to catch
unintentional numerical regressions.

**When to update pins:** only when you intentionally change a fitting algorithm
and the new output is correct (e.g., a bug fix that shifts parameter estimates,
or a numerical improvement).

**How to update pins:**

```bash
# Regenerate all golden files
python -m pytest tests/test_golden.py --regen-goldens

# Review the diff carefully — every changed value should be explainable
git diff tests/golden/

# Add and commit the updated files along with your algorithm change
git add tests/golden/
```

Never update golden pins without a corresponding algorithmic change; a pure
pin-update commit is a red flag during review.

---

## Reporting bugs

Please use the [Bug Report issue template](.github/ISSUE_TEMPLATE/bug_report.md).
Include a minimal reproducible example — the smaller, the better. Paste the full
traceback and list your environment (OS, Python version, NumPy version,
cylinderfit2026 version).

---

## Code of conduct

We are committed to providing a welcoming and respectful environment for everyone,
regardless of experience level, gender, identity, background, or affiliation.

**In short:** be kind, assume good intent, give constructive feedback, and focus
on the work — not the person. Harassment of any kind will not be tolerated.

If you experience or witness unacceptable behaviour, please open a confidential
report by emailing the maintainers directly (address in `pyproject.toml`).
