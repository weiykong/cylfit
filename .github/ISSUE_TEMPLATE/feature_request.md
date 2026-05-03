---
name: Feature request
about: Propose a new feature or enhancement for cylinderfit2026
labels: enhancement
assignees: ''
---

## Is your feature request related to a problem?

A clear and concise description of what problem you are trying to solve.
For example: "When fitting a pipe network with noisy data, there is no way to
specify per-segment noise tolerance, so the global threshold degrades results
on heterogeneous point clouds."

## Describe the solution you would like

A clear and concise description of what you want to happen. If you have ideas
about the API surface (function signatures, class names, configuration options),
sketch them out here:

```python
# Proposed API sketch (optional)
result = cf.fit_pipe_network(
    points,
    per_segment_noise=True,
    noise_estimator="median_absolute_deviation",
)
```

## Describe alternatives you have considered

Other approaches, workarounds, or related libraries you have looked at, and
why they do not fully address the need.

## Additional context

Relevant papers, links, benchmarks, example datasets, or any other information
that would help evaluate or implement this request.
