# SAFE-AI Metrics

SAFE-AI Metrics is a Python package for evaluating machine learning models with accuracy-, robustness-, and explainability-oriented Rank Graduation metrics.

The package provides a unified public API for:

- **RGA** — Rank Graduation Accuracy
- **RGR** — Rank Graduation Robustness
- **RGE** — Rank Graduation Explainability

## Installation

Install from PyPI:

```bash
pip install safe-ai-metrics
```

## Documentation

Full documentation is available on ReadTheDocs:

https://safeai.readthedocs.io/

If your ReadTheDocs project slug is different, replace the link above with the URL shown by your ReadTheDocs project.

## Quickstart

```python
from safeai.rga import rga_score, aurga_score, rga_curve, compare_rga
from safeai.rgr import rgr_score, aurgr_score, rgr_curve, compare_rgr
from safeai.rge import rge_score, aurge_score, rge_curve, compare_rge
```

Use `*_score(...)` for one scalar value, `aur*_score(...)` for only the area under a curve, `*_curve(...)` for the full curve result, and `compare_*(...)` for comparing several models.

## Basic usage

### Rank Graduation Accuracy

Compute a single RGA value:

```python
from safeai.rga import rga_score

score = rga_score(
    y_true,
    y_score
)

print(score)
```

Compute only AURGA:

```python
from safeai.rga import aurga_score

aurga = aurga_score(
    y_true,
    y_score,
    n_segments=10,
    curve_method='auto'
)

print(aurga)
```

Get the full RGA curve result:

```python
from safeai.rga import rga_curve

result = rga_curve(
    y_true,
    y_score,
    n_segments=10,
    curve_method='auto'
)

print(result['rga'])
print(result['aurga'])
print(result['curve'])
```

Compare several models or probability arrays:

```python
from safeai.rga import compare_rga

results = compare_rga(
    {
        'Model A': y_score_a,
        'Model B': y_score_b
    },
    y_true,
    n_segments=10
)
```

### Rank Graduation Robustness

Compute a single RGR value between original and perturbed predictions:

```python
from safeai.rgr import rgr_score

score = rgr_score(
    pred_original,
    pred_perturbed,
    class_order=[0, 1]
)

print(score)
```

Compute only AURGR for a robustness curve:

```python
from safeai.rgr import aurgr_score

aurgr = aurgr_score(
    model,
    x_data,
    strengths=[0.0, 0.05, 0.10],
    method='noise',
    prob_original=prob_original,
    model_class_order=model.classes_,
    class_order=[0, 1]
)

print(aurgr)
```

Get the full RGR curve result:

```python
from safeai.rgr import rgr_curve

result = rgr_curve(
    model,
    x_data,
    strengths=[0.0, 0.05, 0.10],
    method='noise',
    prob_original=prob_original,
    model_class_order=model.classes_,
    class_order=[0, 1]
)

print(result['aurgr'])
print(result['rgr_scores'])
```

Compare several models:

```python
from safeai.rgr import compare_rgr

results = compare_rgr(
    {
        'Model A': (model_a, x_data, prob_a, model_a.classes_, 'sklearn', None),
        'Model B': (model_b, x_data, prob_b, model_b.classes_, 'sklearn', None),
    },
    strengths=[0.0, 0.05, 0.10],
    class_order=[0, 1],
    method='noise'
)
```

### Rank Graduation Explainability

Compute a single RGE value between full and reduced predictions:

```python
from safeai.rge import rge_score

score = rge_score(
    pred_full,
    pred_reduced,
    class_order=[0, 1]
)

print(score)
```

Compute only AURGE for a feature-removal curve:

```python
from safeai.rge import aurge_score

aurge = aurge_score(
    model,
    x_data,
    method='tabular',
    feature_names=feature_names,
    model_class_order=model.classes_,
    class_order=[0, 1],
    n_steps=10
)

print(aurge)
```

Get the full RGE curve result:

```python
from safeai.rge import rge_curve

result = rge_curve(
    model,
    x_data,
    method='tabular',
    feature_names=feature_names,
    model_class_order=model.classes_,
    class_order=[0, 1],
    n_steps=10
)

print(result['aurge'])
print(result['rge_scores'])
```

Compare several models:

```python
from safeai.rge import compare_rge

results = compare_rge(
    {
        'Model A': (model_a, x_data, feature_names, prob_a, model_a.classes_, 'sklearn', None),
        'Model B': (model_b, x_data, feature_names, prob_b, model_b.classes_, 'sklearn', None),
    },
    class_order=[0, 1],
    method='tabular',
    n_steps=10
)
```

## Main API

The main public functions are:

### RGA

- `rga_score`
- `rga_curve`
- `aurga_score`
- `compare_rga`
- `plot_rga`

### RGR

- `rgr_score`
- `rgr_curve`
- `aurgr_score`
- `compare_rgr`
- `plot_rgr`

### RGE

- `rge_score`
- `rge_curve`
- `aurge_score`
- `compare_rge`
- `plot_rge`

## Package structure

The main modules are:

- `safeai.rga` — Rank Graduation Accuracy
- `safeai.rgr` — Rank Graduation Robustness
- `safeai.rge` — Rank Graduation Explainability
- `safeai.cramer` — Lorenz/concordance Cramer-von Mises utilities
- `safeai.utils` — shared utility functions

## Acknowledgements

The development of this package builds on the `safeaipackage` project by Golnoosh Babaei.

The original `safeaipackage` repository is available at:

https://github.com/GolnooshBabaei/safeaipackage

This repository is currently maintained as a separate implementation for development and packaging purposes, but it is expected to be merged or aligned with the original SAFE-AI package in the future.

## Citation

If you use this package in academic work, please cite the SAFE-AI metrics paper:

```bibtex
@article{safeaimetrics,
  title = {{SAFE AI metrics: An integrated approach}},
  journal = {Machine Learning with Applications},
  volume = {23},
  pages = {100821},
  year = {2026},
  issn = {2666-8270},
  doi = {10.1016/j.mlwa.2025.100821},
  url = {https://www.sciencedirect.com/science/article/pii/S266682702500204X},
  author = {Giudici, Paolo and Kolesnikov, Vasily}
}
```

The package is also related to the Rank Graduation Box framework. Please also consider citing:

```bibtex
@article{babaei2025rgb,
  title = {{A Rank Graduation Box for SAFE AI}},
  journal = {Expert Systems with Applications},
  volume = {259},
  pages = {125239},
  year = {2025},
  doi = {10.1016/j.eswa.2024.125239},
  author = {Babaei, Golnoosh and Giudici, Paolo and Raffinetti, Emanuela}
}
```