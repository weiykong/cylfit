---
name: Bug report
about: Report a reproducible defect in cylfit
labels: bug
assignees: ''
---

## Describe the bug

A clear and concise description of what the bug is.

## Minimal reproducible example

Paste the smallest possible snippet that demonstrates the issue. Attach or
inline any point-cloud data required to reproduce it (prefer a synthetic
example generated in code).

```python
import numpy as np
import cylfit as cf

# Minimal code that triggers the bug
rng = np.random.default_rng(0)
points = ...
result = cf.fit_cylinder(points)
print(result)
```

## Expected behaviour

What you expected to happen.

## Actual behaviour

What actually happened. Include the full traceback if an exception was raised:

```
Traceback (most recent call last):
  ...
```

## Environment

| Field | Value |
|-------|-------|
| Operating system | e.g. Ubuntu 22.04 / macOS 15.3 / Windows 11 |
| Python version | e.g. 3.11.9 |
| NumPy version | e.g. 2.1.0 |
| SciPy version | e.g. 1.13.0 |
| cylfit version | e.g. 0.1.0 |
| Installation method | pip / conda / source |

## Additional context

Any other context, related issues, or screenshots that might help.
