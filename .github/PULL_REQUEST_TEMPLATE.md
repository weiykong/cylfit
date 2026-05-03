## Summary

<!-- Explain *what* this PR changes and *why*. Link to any related issue(s).
     e.g. "Fixes the analytic Jacobian for the cone fitter when the half-angle
     approaches zero (#42). The previous code divided by sin(alpha) without
     guarding against alpha ≈ 0, causing NaN values to propagate into the LM
     step." -->

Closes #

## Type of change

- [ ] Bug fix (non-breaking change that resolves an issue)
- [ ] New feature (non-breaking addition of functionality)
- [ ] Breaking change (fix or feature that would cause existing behaviour to change)
- [ ] Refactor (internal restructuring, no behaviour change)
- [ ] Documentation update
- [ ] Test improvement (new or updated tests, no production code change)
- [ ] CI / build / tooling change

## Checklist

- [ ] All tests pass locally: `python -m pytest tests/ -q`
- [ ] `ruff check src/ tests/` exits cleanly (zero issues)
- [ ] `ruff format --check src/ tests/` exits cleanly
- [ ] `mypy src/cylfit` exits cleanly
- [ ] Golden-value pins updated if any fitting algorithm changed
      (`python -m pytest tests/test_golden.py --regen-goldens` and reviewed diff)
- [ ] New public API surfaces have NumPy-style docstrings
- [ ] Relevant documentation updated (docstrings, Sphinx pages, examples)
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]` describing this change
- [ ] PR title follows Conventional Commits style
      (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:`, `perf:`, `chore:`)
