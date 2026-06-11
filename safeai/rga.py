"""
Rank Graduation Accuracy (RGA).

Main functions
--------------
rga_score
    Compute one RGA value.

rga_curve
    Compute an RGA curve and normalized AURGA.

aurga_score
    Compute only the normalized area under the RGA curve.

compare_rga
    Compare several models/probability arrays using RGA curves.

plot_rga
    Plot one RGA curve or a comparison of several RGA curves.

Notes
-----
Notes
-----
The scalar RGA score is computed consistently across binary and multiclass
classification.

AURGA depends on the curve-construction method.

For binary and multiclass classification, the default curve method is
'removal'. This progressively removes the most confident samples inside each
true class and recomputes RGA on the remaining data.

The binary 'partial' curve is still available with curve_method='partial'. It
is useful as a decomposition-style curve, but the removal curve is preferred
for normalized AURGA.
"""

from typing import Any

import numpy as np

from safeai.cramer import gini_via_lorenz, cvm1_concordance_weighted
from safeai.utils import (
    ensure_prob_matrix,
    fill_nan_tail,
    aurga_from_curve,
    ideal_prob_matrix,
    get_model_probabilities
)

__all__ = [
    'rga_score',
    'rga_curve',
    'aurga_score',
    'compare_rga',
    'plot_rga'
]


# ---- Public API ----
def rga_score(
    y_true,
    y_score,
    *,
    x=None,
    class_order=None,
    positive_class=1,
    verbose=False
):
    """
    Compute Rank Graduation Accuracy.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.

    y_score : array-like or fitted model
        One of:
        - binary score/probability vector, shape (n_samples,)
        - probability matrix, shape (n_samples, n_classes)
        - fitted sklearn estimator or Pipeline with predict_proba

    x : array-like or DataFrame, optional
        Input features. Required if y_score is a fitted model.

    class_order : array-like, optional
        Order of probability columns. For sklearn models this is usually
        ``model.classes_``.

    positive_class : int or str, default=1
        Positive class for binary classification.

    verbose : bool, default=False
        Whether to print additional information.

    Returns
    -------
    float
        RGA score.
    """
    y_true = np.asarray(y_true)

    scores, classes, task = _prepare_scores(
        y_score,
        x=x,
        y_true=y_true,
        class_order=class_order,
        positive_class=positive_class
    )

    if task == 'binary':
        return _binary_rga_score(y_true, scores)

    result = _multiclass_rga_score(
        y_true,
        scores,
        class_order=classes,
        verbose=verbose
    )

    return result['rga']


def rga_curve(
    y_true,
    y_score,
    *,
    x=None,
    class_order=None,
    positive_class=1,
    n_segments=10,
    curve_method='auto',
    normalize_to_perfect=True,
    verbose=False
):
    """
    Compute an RGA curve and normalized AURGA.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.

    y_score : array-like or fitted model
        One of:
        - binary score/probability vector, shape (n_samples,)
        - probability matrix, shape (n_samples, n_classes)
        - fitted sklearn estimator or Pipeline with predict_proba

    x : array-like or DataFrame, optional
        Input features. Required if y_score is a fitted model.

    class_order : array-like, optional
        Order of probability columns.

    positive_class : int or str, default=1
        Positive class for binary classification.

    n_segments : int, default=10
        Number of curve segments.

    curve_method : {'auto', 'partial', 'removal'}, default='auto'
        Method used to construct the RGA curve.

        - 'auto':
            binary -> 'partial'
            multiclass -> 'removal'

        - 'partial':
            Partial contribution decomposition.
            Currently supported only for binary classification.

        - 'removal':
            Progressively remove high-confidence samples and recompute RGA.

    normalize_to_perfect : bool, default=True
        If True, return normalized AURGA as the main 'aurga'.

    verbose : bool, default=False
        Whether to print additional information.

    Returns
    -------
    dict
        Dictionary containing:
        - task
        - curve_method
        - rga
        - x
        - curve
        - aurga
        - aurga_raw
        - aurga_perfect, when available
        - perfect_curve, when available
        - per_class_rga, only for multiclass
        - class_weights, only for multiclass
        - classes, only for multiclass
    """
    _validate_n_segments(n_segments)
    _validate_curve_method(curve_method)

    y_true = np.asarray(y_true)

    scores, classes, task = _prepare_scores(
        y_score,
        x=x,
        y_true=y_true,
        class_order=class_order,
        positive_class=positive_class
    )

    resolved_method = _resolve_curve_method(curve_method)

    if task == 'binary':
        if resolved_method == 'partial':
            return _binary_rga_curve_partial(
                y_true,
                scores,
                n_segments=n_segments,
                normalize_to_perfect=normalize_to_perfect
            )

        return _binary_rga_curve_removal(
            y_true,
            scores,
            n_segments=n_segments,
            normalize_to_perfect=normalize_to_perfect
        )

    if resolved_method == 'partial':
        raise ValueError(
            "curve_method='partial' is currently supported only for "
            "binary classification."
        )

    return _multiclass_rga_curve_removal(
        y_true,
        scores,
        class_order=classes,
        n_segments=n_segments,
        normalize_to_perfect=normalize_to_perfect,
        verbose=verbose
    )


def aurga_score(
    y_true,
    y_score,
    *,
    x=None,
    class_order=None,
    positive_class=1,
    n_segments=10,
    curve_method='auto',
    normalize_to_perfect=True,
    verbose=False
):
    """
    Compute only the area under the RGA curve.

    By default, this returns normalized AURGA.
    """
    result = rga_curve(
        y_true,
        y_score,
        x=x,
        class_order=class_order,
        positive_class=positive_class,
        n_segments=n_segments,
        curve_method=curve_method,
        normalize_to_perfect=normalize_to_perfect,
        verbose=verbose
    )

    return result['aurga']


def compare_rga(
    models,
    y_true,
    *,
    x=None,
    n_segments=10,
    positive_class=1,
    curve_method='auto',
    normalize_to_perfect=True,
    save_path=None,
    show=False,
    verbose=True
):
    """
    Compare multiple models or probability arrays using RGA curves.

    Parameters
    ----------
    models : dict
        Dictionary where values can be:
        - score array
        - probability matrix
        - fitted model with predict_proba
        - tuple of (probability_matrix, class_order)

    y_true : array-like
        True labels.

    x : array-like or DataFrame, optional
        Required if model objects are passed.

    n_segments : int, default=10
        Number of curve segments.

    positive_class : int or str, default=1
        Positive class for binary classification.

    curve_method : {'auto', 'partial', 'removal'}, default='auto'
        Method used to construct RGA curves.

    normalize_to_perfect : bool, default=True
        If True, return normalized AURGA as the main 'aurga' value.

    save_path : str or None, default=None
        If provided, save the comparison plot to this path.

    show : bool, default=False
        If True, display the comparison plot with plt.show().

    verbose : bool, default=True
        Whether to print summary.

    Returns
    -------
    dict
        Mapping from model name to RGA curve result.
    """
    results = {}

    for model_name, model_or_scores in models.items():
        if verbose:
            print(f'Evaluating {model_name}...')

        scores, class_order = _unpack_model_value(model_or_scores)

        result = rga_curve(
            y_true,
            scores,
            x=x,
            class_order=class_order,
            positive_class=positive_class,
            n_segments=n_segments,
            curve_method=curve_method,
            normalize_to_perfect=normalize_to_perfect,
            verbose=False
        )

        results[model_name] = result

        if verbose:
            rga = result.get('rga', np.nan)
            aurga = result.get('aurga', np.nan)
            aurga_raw = result.get('aurga_raw', np.nan)
            method = result.get('curve_method', 'unknown')

            print(
                f'{model_name}: '
                f'RGA={rga:.4f}, AURGA={aurga:.4f}, '
                f'AURGA_raw={aurga_raw:.4f}, method={method}'
            )

    if save_path is not None or show:
        plot_rga(results, save_path=save_path, show=show)

    return results


def plot_rga(
    result,
    *,
    model_name='Model',
    fig_size=(12, 5),
    save_path=None,
    show=False
):
    """
    Plot one RGA curve or several RGA curves.

    Parameters
    ----------
    result : dict
        Either:
        - result from rga_curve(...)
        - results from compare_rga(...)

    model_name : str, default='Model'
        Used only when plotting one curve.

    fig_size : tuple, default=(12, 5)
        Figure size.

    save_path : str or None, default=None
        If provided, save the plot.

    show : bool, default=False
        If True, display the plot with plt.show().

    Returns
    -------
    str or tuple
        If save_path is provided and show is False, returns save_path.
        Otherwise, returns (fig, ax).
    """
    if save_path is not None and not show:
        import matplotlib
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    if _is_single_rga_result(result):
        results = {model_name: result}
        title = 'RGA Curve'
    else:
        results = result
        title = 'RGA Curves Comparison'

    fig, ax = plt.subplots(figsize=fig_size)

    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (name, res), color in zip(results.items(), colors):
        ax.plot(
            res['x'],
            res['curve'],
            '-o',
            linewidth=2.3,
            markersize=4.5,
            color=color,
            label=_curve_label(name, res)
        )

        if len(results) == 1 and 'perfect_curve' in res:
            ax.plot(
                res['x'],
                res['perfect_curve'],
                '--',
                linewidth=2.0,
                color=color,
                alpha=0.6,
                label='Perfect'
            )

    ax.set_xlabel('Fraction of Data Removed', fontsize=11, fontweight='bold')
    ax.set_ylabel('RGA Score', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.legend(fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
        return fig, ax

    if save_path is not None:
        plt.close(fig)
        return save_path

    return fig, ax


# ---- Input handling helpers ----
def _prepare_scores(
    y_score,
    *,
    x=None,
    y_true=None,
    class_order=None,
    positive_class=1
):
    """
    Convert user input into a clean score representation.

    Returns
    -------
    scores : np.ndarray
        Either:
        - binary score vector, shape (n_samples,)
        - multiclass probability matrix, shape (n_samples, n_classes)

    classes : np.ndarray or None
        Class order for multiclass.

    task : str
        Either 'binary' or 'multiclass'.
    """
    model = y_score

    if hasattr(model, 'predict_proba'):
        if x is None:
            raise ValueError('x must be provided when y_score is a fitted model.')

        probabilities = get_model_probabilities(model, x)

        if class_order is None and hasattr(model, 'classes_'):
            class_order = np.asarray(model.classes_)

        y_score = probabilities

    y_score = np.asarray(y_score)

    if y_score.ndim == 1:
        _, clean_score = _clean_binary_inputs(y_true, y_score)
        return clean_score, None, 'binary'

    if y_score.ndim != 2:
        raise ValueError(
            'y_score must be either a 1D score vector, '
            'a 2D probability matrix, or a fitted model with predict_proba.'
        )

    if class_order is None:
        if y_true is None:
            raise ValueError(
                'class_order must be provided for probability matrices '
                'when y_true is not available.'
            )
        class_order = np.unique(y_true)
    else:
        class_order = np.asarray(class_order)

    y_score = ensure_prob_matrix(y_score, class_order)

    if y_score.shape[1] != len(class_order):
        raise ValueError(
            f'Probability matrix has {y_score.shape[1]} columns, '
            f'but class_order has {len(class_order)} classes.'
        )

    if y_score.shape[1] == 2:
        classes_list = list(class_order)

        if positive_class in classes_list:
            pos_idx = classes_list.index(positive_class)
        else:
            pos_idx = 1

        _, clean_score = _clean_binary_inputs(y_true, y_score[:, pos_idx])
        return clean_score, class_order, 'binary'

    _, clean_prob = _clean_multiclass_inputs(y_true, y_score)
    return clean_prob, class_order, 'multiclass'


def _unpack_model_value(value):
    """
    Allow compare_rga values to be either:
    - scores
    - model
    - (scores, class_order)
    """
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]

    return value, None


def _clean_binary_inputs(y_true, y_score):
    """
    Validate and clean binary inputs.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)

    if len(y_true) != len(y_score):
        raise ValueError(
            f'y_true and y_score must have the same length. '
            f'Got {len(y_true)} and {len(y_score)}.'
        )

    mask = np.isfinite(y_true) & np.isfinite(y_score)

    return y_true[mask], y_score[mask]


def _clean_multiclass_inputs(y_true, prob_matrix):
    """
    Validate and clean multiclass inputs.
    """
    y_true = np.asarray(y_true).reshape(-1)
    prob_matrix = np.asarray(prob_matrix, dtype=float)

    if len(y_true) != prob_matrix.shape[0]:
        raise ValueError(
            f'y_true and prob_matrix must have the same number of rows. '
            f'Got {len(y_true)} and {prob_matrix.shape[0]}.'
        )

    mask = np.isfinite(prob_matrix).all(axis=1)

    return y_true[mask], prob_matrix[mask]


# ---- Curve method helpers ----
def _validate_n_segments(n_segments):
    """
    Validate number of segments.
    """
    if not isinstance(n_segments, int):
        raise TypeError('n_segments must be an integer.')

    if n_segments < 1:
        raise ValueError('n_segments must be at least 1.')


def _validate_curve_method(curve_method):
    """
    Validate curve method.
    """
    valid_methods = {'auto', 'partial', 'removal'}

    if curve_method not in valid_methods:
        raise ValueError(
            f'curve_method must be one of {valid_methods}. '
            f'Got {curve_method}.'
        )


def _resolve_curve_method(curve_method):
    """
    Resolve automatic curve method.

    By default, both binary and multiclass classification use the removal
    curve. The binary partial curve remains available explicitly with
    curve_method='partial'.
    """
    if curve_method != 'auto':
        return curve_method

    return 'removal'


# ---- Binary helpers ----
def _binary_rga_score(y_true, y_score):
    """
    Binary or numeric RGA based on CvM distance.

    RGA = 1 - CvM(y_true, y_score) / Gini(y_true)
    """
    y_true, y_score = _clean_binary_inputs(y_true, y_score)

    if len(y_true) == 0:
        return np.nan

    gini = gini_via_lorenz(y_true)

    if not np.isfinite(gini) or gini == 0:
        return np.nan

    cvm = cvm1_concordance_weighted(y_true, y_score)

    if not np.isfinite(cvm):
        return np.nan

    return float(1 - cvm / gini)


def _binary_rga_curve_partial(
    y_true,
    y_score,
    *,
    n_segments,
    normalize_to_perfect=True
):
    """
    Binary RGA curve using partial RGA contribution decomposition.

    This is the default binary curve because it is smoother and more stable
    than removal-based recomputation for binary classification.
    """
    y_true, y_score = _clean_binary_inputs(y_true, y_score)

    x_axis = np.linspace(0, 1, n_segments + 1)
    n = len(y_true)

    if n == 0:
        return {
            'task': 'binary',
            'curve_method': 'partial',
            'rga': np.nan,
            'x': x_axis,
            'curve': np.full(n_segments + 1, np.nan),
            'partial': np.full(n_segments, np.nan),
            'aurga': np.nan,
            'aurga_raw': np.nan,
            'segment_indices': []
        }

    full_rga = _binary_rga_score(y_true, y_score)
    full_gini = gini_via_lorenz(y_true)

    if not np.isfinite(full_rga) or not np.isfinite(full_gini) or full_gini == 0:
        return {
            'task': 'binary',
            'curve_method': 'partial',
            'rga': full_rga,
            'x': x_axis,
            'curve': np.full(n_segments + 1, np.nan),
            'partial': np.full(n_segments, np.nan),
            'aurga': np.nan,
            'aurga_raw': np.nan,
            'segment_indices': []
        }

    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    score_sorted = y_score[order]

    segments = _make_segments(n, n_segments)

    partial = []
    segment_indices = []

    for start, end in segments:
        segment_indices.append((start, end))

        y_segment = y_sorted[start:end]
        score_segment = score_sorted[start:end]

        segment_rga = _binary_rga_score(y_segment, score_segment)
        segment_gini = gini_via_lorenz(y_segment)

        if (
            np.isfinite(segment_rga)
            and np.isfinite(segment_gini)
            and segment_gini > 0
        ):
            weight = len(y_segment) / n
            contribution = segment_rga * segment_gini * weight / full_gini
        else:
            contribution = 0.0

        partial.append(contribution)

    partial = np.asarray(partial, dtype=float)
    partial_sum = np.sum(partial)

    if partial_sum > 0:
        partial = partial * (full_rga / partial_sum)

    curve = np.zeros(n_segments + 1, dtype=float)
    curve[0] = full_rga

    removed = 0.0
    for i in range(n_segments):
        removed += partial[i]
        curve[i + 1] = full_rga - removed

    curve = fill_nan_tail(curve)
    aurga_raw = float(aurga_from_curve(curve))

    result = {
        'task': 'binary',
        'curve_method': 'partial',
        'rga': float(full_rga),
        'x': x_axis,
        'curve': curve,
        'partial': partial,
        'aurga': aurga_raw,
        'aurga_raw': aurga_raw,
        'segment_indices': segment_indices
    }

    if normalize_to_perfect:
        perfect_scores = _ideal_binary_scores(y_true)

        perfect_result: dict[str, Any] = _binary_rga_curve_partial(
            y_true,
            perfect_scores,
            n_segments=n_segments,
            normalize_to_perfect=False
        )

        perfect_curve = perfect_result['curve']
        perfect_aurga = float(perfect_result['aurga_raw'])
        aurga_normalized = _safe_normalize_area(aurga_raw, perfect_aurga)

        result['perfect_curve'] = perfect_curve
        result['aurga_perfect'] = perfect_aurga
        result['aurga'] = aurga_normalized

    return result


def _binary_rga_curve_removal(
    y_true,
    y_score,
    *,
    n_segments,
    normalize_to_perfect=True
):
    """
    Binary RGA curve by class-wise removal and recomputation.

    This mirrors the multiclass removal logic. At each step, the same fraction
    of samples is removed inside each true class. This keeps the perfect-model
    normalization stable.
    """
    y_true, y_score = _clean_binary_inputs(y_true, y_score)

    x_axis = np.linspace(0, 1, n_segments + 1)
    curve = np.zeros_like(x_axis, dtype=float)

    full_rga = _binary_rga_score(y_true, y_score)

    classes = np.unique(y_true)

    if len(y_true) == 0 or len(classes) < 2:
        curve[:] = np.nan
        aurga_raw = np.nan
    else:
        idx_by_class = {
            cls: np.where(y_true == cls)[0]
            for cls in classes
        }

        positive_class = np.max(classes)

        for i, frac in enumerate(x_axis):
            keep_indices = []

            for cls in classes:
                idx_cls = idx_by_class[cls]

                if len(idx_cls) == 0:
                    continue

                if cls == positive_class:
                    confidence = y_score[idx_cls]
                else:
                    confidence = -y_score[idx_cls]

                order_cls = idx_cls[np.lexsort((idx_cls, -confidence))]

                n_remove = int(np.floor(frac * len(idx_cls)))
                keep_cls = order_cls[n_remove:]

                keep_indices.append(keep_cls)

            if keep_indices:
                keep_indices = np.concatenate(keep_indices)
            else:
                keep_indices = np.array([], dtype=int)

            if len(keep_indices) < 2 or len(np.unique(y_true[keep_indices])) < 2:
                curve[i] = 0.0
                continue

            value = _binary_rga_score(
                y_true[keep_indices],
                y_score[keep_indices]
            )

            curve[i] = float(value) if np.isfinite(value) else 0.0

        curve = fill_nan_tail(curve)
        aurga_raw = float(aurga_from_curve(curve))

    result = {
        'task': 'binary',
        'curve_method': 'removal',
        'rga': full_rga,
        'x': x_axis,
        'curve': curve,
        'aurga': aurga_raw,
        'aurga_raw': aurga_raw
    }

    if normalize_to_perfect:
        perfect_scores = _ideal_binary_scores(y_true)

        perfect_result: dict[str, Any] = _binary_rga_curve_removal(
            y_true,
            perfect_scores,
            n_segments=n_segments,
            normalize_to_perfect=False
        )

        perfect_curve = perfect_result['curve']
        perfect_aurga = float(perfect_result['aurga_raw'])
        aurga_normalized = _safe_normalize_area(aurga_raw, perfect_aurga)

        result['perfect_curve'] = perfect_curve
        result['aurga_perfect'] = perfect_aurga
        result['aurga'] = aurga_normalized

    return result


def _ideal_binary_scores(y_true):
    """
    Ideal binary scores.

    For binary labels this is simply y_true itself. This means positive
    observations receive the highest score.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    return y_true.copy()


def _make_segments(n, n_segments):
    """
    Make approximately equal index segments.
    """
    segment_size = n // n_segments
    remainder = n % n_segments

    segments = []
    start = 0

    for i in range(n_segments):
        current_size = segment_size + (1 if i < remainder else 0)
        end = start + current_size
        segments.append((start, end))
        start = end

    return segments


# ---- Multiclass helpers ----
def _multiclass_rga_score(
    y_true,
    prob_matrix,
    *,
    class_order,
    verbose=False
):
    """
    Weighted one-vs-rest multiclass RGA.
    """
    y_true, prob_matrix = _clean_multiclass_inputs(y_true, prob_matrix)
    class_order = np.asarray(class_order)

    rgas = []
    weights = []

    for k, cls in enumerate(class_order):
        y_binary = np.equal(y_true, cls).astype(float)
        score_cls = prob_matrix[:, k]

        if np.sum(y_binary) == 0:
            if verbose:
                print(f'Warning: class {cls} has no samples. Skipping.')
            rgas.append(np.nan)
            weights.append(0.0)
            continue

        rga_cls = _binary_rga_score(y_binary, score_cls)

        rgas.append(rga_cls)
        weights.append(np.mean(y_binary))

    rgas = np.asarray(rgas, dtype=float)
    weights = np.asarray(weights, dtype=float)

    denom = np.nansum(weights)

    if denom > 0:
        weighted_rga = np.nansum(rgas * weights) / denom
    else:
        weighted_rga = np.nan

    return {
        'rga': float(weighted_rga) if np.isfinite(weighted_rga) else np.nan,
        'per_class_rga': rgas,
        'class_weights': weights,
        'classes': class_order
    }


def _multiclass_rga_curve_removal(
    y_true,
    prob_matrix,
    *,
    class_order,
    n_segments,
    normalize_to_perfect=True,
    verbose=False
):
    """
    Multiclass RGA curve by class-wise removal and recomputation.

    At each step:
    1. For every true class, rank its samples by predicted probability
       for that class.
    2. Remove the most confident samples inside each class.
    3. Recompute weighted one-vs-rest RGA on the remaining data.
    """
    y_true, prob_matrix = _clean_multiclass_inputs(y_true, prob_matrix)
    class_order = np.asarray(class_order)

    x_axis = np.linspace(0, 1, n_segments + 1)
    curve = np.zeros_like(x_axis, dtype=float)

    full = _multiclass_rga_score(
        y_true,
        prob_matrix,
        class_order=class_order,
        verbose=verbose
    )

    idx_by_class = {
        cls: np.where(y_true == cls)[0]
        for cls in class_order
    }

    for i, frac in enumerate(x_axis):
        keep_indices = []

        for k, cls in enumerate(class_order):
            idx_cls = idx_by_class[cls]

            if len(idx_cls) == 0:
                continue

            confidence = prob_matrix[idx_cls, k]

            order_cls = idx_cls[np.lexsort((idx_cls, -confidence))]

            n_remove = int(np.floor(frac * len(idx_cls)))
            keep_cls = order_cls[n_remove:]

            keep_indices.append(keep_cls)

        if keep_indices:
            keep_indices = np.concatenate(keep_indices)
        else:
            keep_indices = np.array([], dtype=int)

        if len(keep_indices) < 2:
            curve[i] = 0.0
            continue

        y_remaining = y_true[keep_indices]
        p_remaining = prob_matrix[keep_indices, :]

        score_result = _multiclass_rga_score(
            y_remaining,
            p_remaining,
            class_order=class_order,
            verbose=verbose
        )

        value = score_result['rga']
        curve[i] = float(value) if np.isfinite(value) else 0.0

    curve = fill_nan_tail(curve)
    aurga_raw = float(aurga_from_curve(curve))

    result = {
        'task': 'multiclass',
        'curve_method': 'removal',
        'rga': full['rga'],
        'x': x_axis,
        'curve': curve,
        'aurga': aurga_raw,
        'aurga_raw': aurga_raw,
        'per_class_rga': full['per_class_rga'],
        'class_weights': full['class_weights'],
        'classes': class_order
    }

    if normalize_to_perfect:
        perfect_probs = ideal_prob_matrix(y_true, class_order)

        perfect_result: dict[str, Any] = _multiclass_rga_curve_removal(
            y_true,
            perfect_probs,
            class_order=class_order,
            n_segments=n_segments,
            normalize_to_perfect=False,
            verbose=False
        )

        perfect_curve = perfect_result['curve']
        perfect_aurga = float(perfect_result['aurga_raw'])
        aurga_normalized = _safe_normalize_area(aurga_raw, perfect_aurga)

        result['perfect_curve'] = perfect_curve
        result['aurga_perfect'] = perfect_aurga
        result['aurga'] = aurga_normalized

    return result


# ---- Private helpers ----
def _is_single_rga_result(result):
    """
    Check whether a dictionary is a single RGA result.
    """
    return isinstance(result, dict) and 'curve' in result and 'x' in result


def _curve_label(model_name, result):
    """
    Make curve label.
    """
    rga = result.get('rga', np.nan)
    aurga = result.get('aurga', np.nan)
    method = result.get('curve_method', None)

    if np.isfinite(aurga):
        return f'{model_name} (RGA={rga:.3f}, AURGA={aurga:.3f}, {method})'

    return f'{model_name} (RGA={rga:.3f}, {method})'


def _safe_normalize_area(aurga_raw, aurga_perfect):
    """
    Normalize AURGA by the perfect baseline area.
    """
    if np.isfinite(aurga_perfect) and aurga_perfect > 0:
        return float(aurga_raw) / float(aurga_perfect)

    return np.nan