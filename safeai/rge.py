"""
Rank Graduation Explainability (RGE).

Main functions
--------------
rge_score
    Compute one RGE value from full and reduced/occluded predictions.

rge_curve
    Compute one RGE curve and AURGE value for one model.

aurge_score
    Compute only the area under an RGE curve.

compare_rge
    Compare several models using one of the supported RGE workflows.

plot_rge
    Plot one RGE curve or a comparison of several RGE curves.

Notes
-----
RGE measures how model predictions change when information is removed or
occluded. In this implementation, higher values mean stronger preservation of
the original predictions under feature/image removal, and lower values mean
stronger degradation.
"""

from typing import Any, Literal, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from safeai.cramer import gini_via_lorenz, cvm1_concordance_weighted
from safeai.utils import (
    apply_patch_occlusion,
    ensure_prob_matrix,
    get_predictions_from_features,
    apply_importance_masking,
    clean_pair,
    validate_method,
    validate_class_weights,
    rescale_by_rga,
    area_under_normalized_curve,
    nan_to_zero,
    resolve_class_orders,
    apply_feature_baseline,
    mask_columns,
    normalize_rankings
)


RGEMethod = Literal['image', 'text', 'tabular']
ImageOcclusionMethod = Literal['random', 'gradcam_most']
FeatureMaskingMethod = Literal['random', 'most_important', 'greedy']
Baseline = Literal['zero', 'mean']


__all__ = [
    'rge_score',
    'rge_curve',
    'aurge_score',
    'compare_rge',
    'plot_rge'
]


# ---- Public API ----
def rge_score(
    pred_full,
    pred_reduced,
    *,
    class_order=None,
    class_weights=None,
    verbose=False
):
    """
    Compute Rank Graduation Explainability between two prediction arrays.

    Parameters
    ----------
    pred_full : array-like
        Predictions from the full/original model input. Can be a 1D score
        vector or a 2D probability matrix.

    pred_reduced : array-like
        Predictions after feature removal, masking, or occlusion. Must have the
        same shape as pred_full.

    class_order : array-like, optional
        Class order. If provided with 1D binary probabilities, the vectors are
        converted to two-column probability matrices.

    class_weights : array-like, optional
        Weights for multiclass aggregation. If None, uses uniform weights.

    verbose : bool, default=False
        Whether to print per-class values for multiclass inputs.

    Returns
    -------
    float
        RGE score. Higher values indicate stronger prediction preservation
        after removal/occlusion.
    """
    pred_full = np.asarray(pred_full)
    pred_reduced = np.asarray(pred_reduced)

    if pred_full.ndim == 1 and pred_reduced.ndim == 1:
        if class_order is None:
            return 1.0 - _rge_cvm_ratio(pred_full, pred_reduced)

        score, _, _ = _rge_cramer_multiclass(
            pred_full,
            pred_reduced,
            class_order=class_order,
            class_weights=class_weights,
            verbose=verbose
        )
        return score

    score, _, _ = _rge_cramer_multiclass(
        pred_full,
        pred_reduced,
        class_order=class_order,
        class_weights=class_weights,
        verbose=verbose
    )
    return score


def rge_curve(
    model,
    data,
    removal_fractions=None,
    *,
    method: RGEMethod = 'tabular',
    preprocess_fn=None,
    feature_names=None,
    model_class_order=None,
    class_order=None,
    model_type='sklearn',
    device=None,
    batch_size=64,
    class_weights=None,
    model_name='Model',
    rga_full=None,
    occlusion_method: ImageOcclusionMethod = 'random',
    masking_method: FeatureMaskingMethod = 'greedy',
    baseline: Baseline = 'zero',
    feature_ranking=None,
    patch_size=32,
    patch_rankings=None,
    patch_meta=None,
    n_steps=None,
    random_seed=None,
    mask_value=0.0,
    prob_full=None,
    plot=False,
    fig_size=(10, 6),
    save_path=None,
    show=False,
    verbose=True
):
    """
    Compute one Rank Graduation Explainability (RGE) curve and AURGE value.

    Parameters
    ----------
    model : object
        Trained sklearn estimator with predict_proba or PyTorch module that
        returns logits.

    data : object
        Input data used for the selected workflow.

        For method='image', pass a PyTorch Dataset or compatible dataset
        returning image tensors.

        For method='text' or method='tabular', pass a feature matrix.

    removal_fractions : array-like, optional
        Fractions of removed information. For image and text workflows this is
        required. For tabular workflow, if None, the curve is built using
        n_steps or all available features.

    method : {'image', 'text', 'tabular'}, default='tabular'
        RGE workflow used to construct the curve.

    preprocess_fn : callable, optional
        Required for method='image'. Maps image tensors to model-ready feature
        matrices.

    feature_names : array-like, optional
        Required for method='tabular'. Names of the feature columns.

    model_class_order : array-like
        Class order produced by the model probability output.

    class_order : array-like
        Target class order used to align probability columns.

    model_type : {'sklearn', 'pytorch'}, default='sklearn'
        Type of model being evaluated.

    device : torch.device or str, optional
        Device used for PyTorch inference.

    batch_size : int, default=64
        Batch size used when loading images or running PyTorch prediction.

    class_weights : array-like, optional
        Weights used to aggregate per-class RGE values. If None, uniform class
        weights are used.

    model_name : str, default='Model'
        Name stored in the result dictionary and used in plot labels.

    rga_full : float, optional
        If provided and finite, the RGE curve is rescaled by this RGA value.

    occlusion_method : {'random', 'gradcam_most'}, default='random'
        Image occlusion workflow used when method='image'.

    masking_method : {'random', 'most_important', 'greedy'}, default='greedy'
        Feature masking workflow used when method='text' or method='tabular'.
        Text supports 'random' and 'most_important'. Tabular supports all three.

    baseline : {'zero', 'mean'}, default='zero'
        Baseline value used when masking text/tabular features.

    feature_ranking : array-like, optional
        Feature ranking required when masking_method='most_important'.

    patch_size : int, default=32
        Patch size used for image occlusion.

    patch_rankings, patch_meta : optional
        Required when occlusion_method='gradcam_most'.

    n_steps : int, optional
        Number of feature-removal steps for method='tabular'. If None, all
        features are removed one by one.

    random_seed : int, optional
        Random seed used for random masking or occlusion.

    mask_value : float, default=0.0
        Constant value used for image patch masking.

    prob_full : array-like, optional
        Cached full/original probability matrix. If None, it is computed.

    plot : bool, default=False
        Whether to create a plot for the computed RGE curve.

    fig_size : tuple, default=(10, 6)
        Figure size used when plotting.

    save_path : str, optional
        Path where the plot should be saved.

    show : bool, default=False
        Whether to display the plot with plt.show().

    verbose : bool, default=True
        Whether to print progress and summary information.

    Returns
    -------
    dict
        Dictionary containing the RGE curve, AURGE, removed fractions, optional
        rescaled curve, per-class RGE values, and method metadata.
    """
    validate_method(method, allowed={'image', 'text', 'tabular'})
    model_class_order, class_order = resolve_class_orders(
        model,
        model_class_order=model_class_order,
        class_order=class_order,
        prob=prob_full
    )

    if method == 'image':
        if preprocess_fn is None:
            raise ValueError("preprocess_fn is required when method='image'.")
        if removal_fractions is None:
            raise ValueError("removal_fractions is required when method='image'.")

        result = _rge_curve_image_core(
            model=model,
            preprocess_fn=preprocess_fn,
            images_dataset=data,
            removal_fractions=removal_fractions,
            model_class_order=model_class_order,
            class_order=class_order,
            model_type=model_type,
            device=device,
            patch_size=patch_size,
            batch_size=batch_size,
            class_weights=class_weights,
            model_name=model_name,
            rga_full=rga_full,
            occlusion_method=occlusion_method,
            patch_rankings=patch_rankings,
            patch_meta=patch_meta,
            plot=False,
            fig_size=fig_size,
            verbose=verbose,
            random_seed=random_seed,
            mask_value=mask_value,
            save_path=None
        )

    elif method == 'text':
        if removal_fractions is None:
            raise ValueError("removal_fractions is required when method='text'.")
        if masking_method == 'greedy':
            raise ValueError("method='text' supports masking_method='random' or 'most_important'.")

        result = _rge_curve_text_core(
            model=model,
            x=data,
            removal_fractions=removal_fractions,
            model_class_order=model_class_order,
            class_order=class_order,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            class_weights=class_weights,
            model_name=model_name,
            rga_full=rga_full,
            masking_method=masking_method,
            feature_ranking=feature_ranking,
            baseline=baseline,
            plot=False,
            fig_size=fig_size,
            verbose=verbose,
            random_seed=random_seed,
            save_path=None,
            prob_full_cached=prob_full
        )

    else:
        if feature_names is None:
            raise ValueError("feature_names is required when method='tabular'.")

        result = _rge_curve_tabular_core(
            model=model,
            x=data,
            feature_names=feature_names,
            model_class_order=model_class_order,
            class_order=class_order,
            model_type=model_type,
            device=device,
            class_weights=class_weights,
            model_name=model_name,
            rga_full=rga_full,
            masking_method=masking_method,
            feature_ranking=feature_ranking,
            baseline=baseline,
            n_steps=n_steps,
            random_seed=random_seed,
            verbose=verbose,
            plot=False,
            fig_size=fig_size,
            save_path=None,
            prob_full_cached=prob_full
        )

    result['method'] = method
    result['model_name'] = model_name

    if plot or save_path is not None or show:
        plot_rge(
            result,
            model_name=model_name,
            fig_size=fig_size,
            save_path=save_path,
            show=show
        )

    return result


def aurge_score(
    model,
    data,
    removal_fractions=None,
    **kwargs
):
    """
    Compute only the area under an RGE curve.
    """
    result = rge_curve(model, data, removal_fractions, plot=False, **kwargs)
    return result['aurge']


def compare_rge(
    models,
    class_order,
    *,
    method: RGEMethod = 'tabular',
    removal_fractions=None,
    images_dataset=None,
    occlusion_method='random',
    patch_size=32,
    batch_size=64,
    class_weights=None,
    rga_dict=None,
    device=None,
    fig_size=(12, 6),
    verbose=True,
    random_seed=None,
    patch_rankings=None,
    patch_meta=None,
    save_path=None,
    show=False,
    mask_value=0.0,
    use_shared_feature_cache=True,
    masking_method='greedy',
    baseline='zero',
    n_steps=None,
    feature_rankings=None
):
    """
    Compare several models using Rank Graduation Explainability (RGE) curves.

    This is the main user-facing comparison function for RGE. It provides one
    unified interface for all supported RGE workflows:

    - Image occlusion, using method='image'
    - Text or generic feature removal, using method='text'
    - Tabular feature removal, using method='tabular'

    The function returns one result dictionary per model and can optionally plot a
    comparison of the resulting RGE curves.

    Parameters
    ----------
    models : dict
        Dictionary containing model configurations.

        For method='image', each entry must have the form::

            model_name -> (
                model,
                preprocess_fn,
                model_class_order,
                model_type
            )

        where:

        - model is a trained sklearn estimator or PyTorch module.
        - preprocess_fn maps image tensors to model-ready feature matrices.
        - model_class_order is the class order of the model probability output.
        - model_type is either 'sklearn' or 'pytorch'.

        For method='text', each entry must have the form::

            model_name -> (
                model,
                x,
                prob_full,
                model_class_order,
                model_type,
                device
            )

        where:

        - x is the feature matrix to mask.
        - prob_full is the original probability matrix, or None.
        - device is the torch device for PyTorch models, or None for sklearn.

        For method='tabular', each entry must have the form::

            model_name -> (
                model,
                x,
                feature_names,
                prob_full,
                model_class_order,
                model_type,
                device
            )

        where feature_names contains the names of the columns in x.

    class_order : array-like
        Shared target class order used to align probability columns across all
        models.

    method : {'image', 'text', 'tabular'}, default='tabular'
        RGE workflow used for all models in the comparison.

    removal_fractions : array-like, optional
        Fractions of removed information.

        Required for method='image' and method='text'.

        For method='tabular', this argument is not used. The tabular curve is
        controlled by n_steps.

    images_dataset : torch.utils.data.Dataset, optional
        Image dataset required when method='image'. The dataset should return image
        tensors, or tuples/lists where the first element is the image tensor.

    occlusion_method : {'random', 'gradcam_most'} or dict, default='random'
        Image occlusion method used when method='image'.

        If a string is provided, the same occlusion method is used for all models.

        If a dictionary is provided, it should map model names to occlusion methods.

    patch_size : int, default=32
        Patch size used for image occlusion.

    batch_size : int, default=64
        Batch size used when loading images and when running PyTorch prediction.

    class_weights : array-like, optional
        Weights used to aggregate per-class RGE values. If None, uniform class
        weights are used.

    rga_dict : dict, optional
        Mapping from model name to full RGA score. If provided, each RGE
        curve is rescaled by the corresponding RGA value.

    device : torch.device or str, optional
        Device used for PyTorch inference.

    fig_size : tuple, default=(12, 6)
        Figure size used for the comparison plot.

    verbose : bool, default=True
        Whether to print progress and summary information.

    random_seed : int, optional
        Random seed used for random image occlusion or random feature masking.

    patch_rankings : list or array-like, optional
        Patch rankings used when occlusion_method='gradcam_most'.

    patch_meta : dict, optional
        Patch metadata used when occlusion_method='gradcam_most'.

    save_path : str, optional
        Path where the comparison plot should be saved. If None and show=False,
        no plot is saved.

    show : bool, default=False
        Whether to display the comparison plot with plt.show().

    mask_value : float, default=0.0
        Constant value used for image patch masking.

    use_shared_feature_cache : bool, default=True
        Whether to cache image features shared across models when method='image'.
        This can speed up comparison when all models use the same preprocessing
        function and occlusion method.

    masking_method : {'random', 'most_important', 'greedy'}, default='greedy'
        Feature masking method used for method='text' and method='tabular'.

        For method='text', only 'random' and 'most_important' are supported.

        For method='tabular', 'random', 'most_important', and 'greedy' are
        supported.

    baseline : {'zero', 'mean'}, default='zero'
        Baseline value used when masking text or tabular features.

    n_steps : int, optional
        Number of feature-removal steps for method='tabular'. If None, all features
        are removed one by one.

    feature_rankings : array-like or dict, optional
        Feature rankings used when masking_method='most_important'.

        If an array is provided, the same ranking is used for all models.

        If a dictionary is provided, it should map model names to feature-ranking
        arrays.

    Returns
    -------
    dict
        Mapping from model name to RGE result dictionary.

        Each result contains:

        - 'rge_scores' : np.ndarray
            Raw RGE scores at each removal level.
        - 'rge_rescaled' : np.ndarray
            RGE scores after optional rescaling by rga_dict.
        - 'aurge' : float
            Area under the RGE curve.
        - 'per_class_rge' : np.ndarray or None
            Per-class RGE values at each removal level.
        - 'class_order' : np.ndarray
            Class order used for probability alignment.
        - 'method' : str
            RGE workflow used in the comparison.

        Depending on method, each result may also include:

        - 'removal_fractions'
        - 'x_axis'
        - 'occlusion_method'
        - 'masking_method'
        - 'baseline'
        - 'removed_features'
    """
    validate_method(method, allowed={'image', 'text', 'tabular'})

    if method == 'image':
        if images_dataset is None:
            raise ValueError("images_dataset is required when method='image'.")
        if removal_fractions is None:
            raise ValueError("removal_fractions is required when method='image'.")

        results = _compare_rge_image_core(
            models=models,
            images_dataset=images_dataset,
            removal_fractions=removal_fractions,
            class_order=class_order,
            occlusion_method=occlusion_method,
            patch_size=patch_size,
            batch_size=batch_size,
            class_weights=class_weights,
            rga_dict=rga_dict,
            device=device,
            verbose=verbose,
            random_seed=random_seed,
            patch_rankings=patch_rankings,
            patch_meta=patch_meta,
            mask_value=mask_value,
            use_shared_feature_cache=use_shared_feature_cache
        )
        x_key = 'removal_fractions'
        x_label = 'Occluded Image Area (%)'
        x_scale = 100.0
        title = 'RGE Curves Comparison'

    elif method == 'text':
        if removal_fractions is None:
            raise ValueError("removal_fractions is required when method='text'.")
        if masking_method == 'greedy':
            raise ValueError("method='text' supports masking_method='random' or 'most_important'.")

        results = _compare_rge_text_core(
            models=models,
            removal_fractions=removal_fractions,
            class_order=class_order,
            masking_method=masking_method,
            baseline=baseline,
            class_weights=class_weights,
            rga_dict=rga_dict,
            batch_size=batch_size,
            verbose=verbose,
            random_seed=random_seed,
            feature_rankings=feature_rankings
        )
        x_key = 'removal_fractions'
        x_label = 'Removed Features (%)'
        x_scale = 100.0
        title = 'RGE Curves Comparison (Text Feature Removal)'

    else:
        results = _compare_rge_tabular_core(
            models=models,
            class_order=class_order,
            class_weights=class_weights,
            rga_dict=rga_dict,
            masking_method=masking_method,
            baseline=baseline,
            n_steps=n_steps,
            verbose=verbose,
            random_seed=random_seed,
            feature_rankings=feature_rankings
        )
        x_key = 'x_axis'
        x_label = 'Fraction of Features Removed'
        x_scale = 1.0
        title = 'RGE Curves Comparison (Tabular Feature Removal)'

    results = cast(dict[str, dict[str, Any]], results)

    for result in results.values():
        result['method'] = method

    if save_path is not None or show:
        plot_rge(
            results,
            x_key=x_key,
            x_label=x_label,
            x_scale=x_scale,
            title=title,
            fig_size=fig_size,
            save_path=save_path,
            show=show
        )

    if verbose:
        _print_comparison_summary(results, metric_name='AURGE')

    return results


def plot_rge(
    result,
    *,
    model_name='Model',
    x_key=None,
    x_label=None,
    x_scale=1.0,
    y_key='rge_rescaled',
    title=None,
    fig_size=(12, 6),
    save_path=None,
    show=False
):
    """
    Plot one RGE curve or a comparison of several RGE curves.

    Parameters
    ----------
    result : dict
        Either one result returned by rge_curve(...) or the full dictionary
        returned by compare_rge(...).

    model_name : str, default='Model'
        Label used when result is a single curve.

    x_key : str, optional
        Name of the x-axis field in each result. If None, it is inferred from
        the result keys.

    x_label : str, optional
        Human-readable x-axis label. If None, a default label is selected.

    x_scale : float, default=1.0
        Multiplier applied to x values before plotting. Use 100 for percentage
        axes when the stored values are fractions.

    y_key : str, default='rge_rescaled'
        Name of the y-axis field to plot.

    title : str, optional
        Plot title. If None, a default title is selected.

    fig_size : tuple, default=(12, 6)
        Matplotlib figure size.

    save_path : str or None, default=None
        If provided, save the plot to this path.

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

    if _is_single_rge_result(result):
        results = {model_name: result}
    else:
        results = result

    if x_key is None:
        first = next(iter(results.values()))
        x_key = _infer_x_key(first)

    if x_label is None:
        x_label = _default_x_label(x_key)

    if title is None:
        title = 'RGE Curve' if len(results) == 1 else 'RGE Curves Comparison'

    fig, ax = plt.subplots(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (name, res), color in zip(results.items(), colors):
        x_values = np.asarray(res[x_key], dtype=float) * float(x_scale)
        y_values = np.asarray(res[y_key], dtype=float)

        ax.plot(
            x_values,
            y_values,
            '-o',
            linewidth=2.3,
            markersize=4.5,
            color=color,
            label=_curve_label(name, res)
        )

    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
    ax.set_ylabel('RGE Score', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
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


# ---- Scalar helpers ----
def _rge_cvm_ratio(pred, pred_reduced):
    """
    Cramer-von Mises ratio used internally by RGE.

    Lower values mean greater similarity. Public RGE score uses 1 - this ratio.
    """
    pred, pred_reduced = clean_pair(pred, pred_reduced)

    if len(pred) == 0:
        return np.nan

    g = gini_via_lorenz(pred)
    if not np.isfinite(g) or g == 0:
        return np.nan

    cvm = cvm1_concordance_weighted(pred, pred_reduced)
    if not np.isfinite(cvm):
        return np.nan

    return float(cvm / g)


def _rge_cramer_multiclass(
    prob_full,
    prob_reduced,
    class_order=None,
    class_weights=None,
    verbose=False
):
    prob_full = np.asarray(prob_full, dtype=float)
    prob_reduced = np.asarray(prob_reduced, dtype=float)

    if class_order is not None:
        class_order = np.asarray(class_order)
        prob_full = ensure_prob_matrix(prob_full, class_order)
        prob_reduced = ensure_prob_matrix(prob_reduced, class_order)
    else:
        if prob_full.ndim != 2 or prob_reduced.ndim != 2:
            raise ValueError('For 1D binary probabilities, pass class_order with 2 classes.')

    if prob_full.shape != prob_reduced.shape:
        raise ValueError(
            f'Shape mismatch: prob_full {prob_full.shape} and '
            f'prob_reduced {prob_reduced.shape}.'
        )

    if prob_full.ndim != 2:
        raise ValueError('prob_full and prob_reduced must be 2D after conversion.')

    n_classes = prob_full.shape[1]
    class_weights = validate_class_weights(class_weights, n_classes)

    rges = []

    for k in range(n_classes):
        rge_k = 1.0 - _rge_cvm_ratio(prob_full[:, k], prob_reduced[:, k])
        rges.append(rge_k)

        if verbose:
            print(f'Class {k}: RGE = {rge_k:.4f}')

    rges = np.asarray(rges, dtype=float)
    valid = np.isfinite(rges) & np.isfinite(class_weights) & (class_weights > 0)

    if np.any(valid):
        rge_weighted = np.sum(rges[valid] * class_weights[valid]) / np.sum(class_weights[valid])
    else:
        rge_weighted = np.nan

    return float(rge_weighted) if np.isfinite(rge_weighted) else np.nan, rges, class_weights


# ---- Image helpers ----
def _rge_curve_image_core(
    model,
    preprocess_fn,
    images_dataset,
    removal_fractions,
    model_class_order,
    class_order,
    model_type='sklearn',
    device=None,
    patch_size=32,
    batch_size=64,
    class_weights=None,
    model_name='Model',
    rga_full=None,
    occlusion_method='random',
    patch_rankings=None,
    patch_meta=None,
    plot=True,
    fig_size=(10, 6),
    verbose=True,
    random_seed=None,
    mask_value=0.0,
    save_path=None
):
    removal_fractions = np.asarray(removal_fractions, dtype=float)

    if occlusion_method in ('gradcam_most', 'gradcam_least'):
        if patch_rankings is None or patch_meta is None:
            raise ValueError('For Grad-CAM masking you must pass patch_rankings and patch_meta')

    if verbose:
        print(f'RGE Evaluation: {model_name}')
        print(f'Occlusion: {occlusion_method}')
        print(f'Testing {len(removal_fractions)} removal fractions')

    images_all = _load_all_images(images_dataset, batch_size=batch_size)
    feat_full = preprocess_fn(images_all)

    prob_full = get_predictions_from_features(
        feat_full,
        model,
        model_class_order,
        class_order,
        model_type=model_type,
        device=device,
        batch_size=batch_size
    )

    rge_scores = []
    per_class_rge_list = []

    for frac in removal_fractions:
        if verbose:
            print(f'\nOcclusion level: {frac * 100:.0f}%')

        images_occ = _build_occluded_images(
            images_all=images_all,
            frac=float(frac),
            occlusion_method=occlusion_method,
            patch_size=patch_size,
            random_seed=random_seed,
            mask_value=mask_value,
            patch_rankings=patch_rankings,
            patch_meta=patch_meta
        )

        feat_occ = preprocess_fn(images_occ)
        prob_occ = get_predictions_from_features(
            feat_occ,
            model,
            model_class_order,
            class_order,
            model_type=model_type,
            device=device,
            batch_size=batch_size
        )

        rge_val, rge_per_class, _ = _rge_cramer_multiclass(
            prob_full,
            prob_occ,
            class_order=class_order,
            class_weights=class_weights
        )

        rge_value = nan_to_zero(rge_val)
        rge_scores.append(rge_value)
        per_class_rge_list.append(rge_per_class)

        if verbose:
            print(f'RGE = {rge_value:.4f}')

    rge_scores = np.asarray(rge_scores, dtype=float)
    per_class_rge_list = np.asarray(per_class_rge_list, dtype=float)
    rge_rescaled = rescale_by_rga(rge_scores, rga_full)
    aurge = area_under_normalized_curve(removal_fractions, rge_rescaled)

    result = {
        'method': 'image',
        'rge_scores': rge_scores,
        'rge_rescaled': rge_rescaled,
        'aurge': aurge,
        'removal_fractions': removal_fractions,
        'per_class_rge': per_class_rge_list,
        'class_order': np.asarray(class_order),
        'occlusion_method': occlusion_method
    }

    if verbose:
        print(f'AURGE: {aurge:.4f}')

    if plot or save_path is not None:
        plot_rge(
            result,
            model_name=model_name,
            x_key='removal_fractions',
            x_label='Occluded Image Area (%)',
            x_scale=100.0,
            title=f'RGE Curve: {model_name} ({occlusion_method})',
            fig_size=fig_size,
            save_path=save_path,
            show=(plot and save_path is None)
        )

    return result


def _load_all_images(images_dataset, batch_size):
    loader = DataLoader(images_dataset, batch_size=batch_size, shuffle=False)
    images_all = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        images_all.append(x)
    return torch.cat(images_all, dim=0)


def _build_occluded_images(
    images_all,
    frac,
    occlusion_method,
    patch_size=32,
    random_seed=None,
    mask_value=0.0,
    patch_rankings=None,
    patch_meta=None
):
    _, _, h, w = images_all.shape
    total_pixels = h * w
    patch_pixels = patch_size * patch_size

    if occlusion_method == 'random':
        pixels_to_remove = int(frac * total_pixels)
        num_patches = pixels_to_remove // patch_pixels
        return apply_patch_occlusion(
            images_all,
            num_patches,
            patch_size,
            random_seed=random_seed,
            mask_value=mask_value
        )

    if occlusion_method == 'gradcam_most':
        return apply_importance_masking(
            images_all,
            patch_rankings,
            patch_meta,
            frac,
            mask_strategy='most_important',
            mask_value=mask_value
        )

    raise ValueError(f'Unknown occlusion_method: {occlusion_method}')


def _precompute_rge_feature_cache(
    preprocess_fn,
    images_dataset,
    removal_fractions,
    batch_size=64,
    occlusion_method='random',
    patch_size=32,
    random_seed=None,
    mask_value=0.0,
    patch_rankings=None,
    patch_meta=None,
    verbose=True
):
    removal_fractions = np.asarray(removal_fractions, dtype=float)

    if occlusion_method in ('gradcam_most', 'gradcam_least'):
        if patch_rankings is None or patch_meta is None:
            raise ValueError('For Grad-CAM masking you must pass patch_rankings and patch_meta')

    if verbose:
        print('Loading images once for shared RGE cache...')

    images_all = _load_all_images(images_dataset, batch_size=batch_size)

    if verbose:
        print('Extracting shared features from original images...')

    feat_full = preprocess_fn(images_all)

    feat_occ_map = {}
    for frac in removal_fractions:
        if verbose:
            print(f'Caching occluded features for {frac * 100:.0f}%')

        images_occ = _build_occluded_images(
            images_all=images_all,
            frac=float(frac),
            occlusion_method=occlusion_method,
            patch_size=patch_size,
            random_seed=random_seed,
            mask_value=mask_value,
            patch_rankings=patch_rankings,
            patch_meta=patch_meta
        )
        feat_occ_map[float(frac)] = preprocess_fn(images_occ)

    return {
        'feat_full': feat_full,
        'feat_occ_map': feat_occ_map,
        'removal_fractions': removal_fractions,
        'occlusion_method': occlusion_method
    }


def _rge_curve_image_cached_core(
    model,
    feature_cache,
    model_class_order,
    class_order,
    model_type='sklearn',
    device=None,
    batch_size=64,
    class_weights=None,
    model_name='Model',
    rga_full=None,
    plot=True,
    fig_size=(10, 6),
    verbose=True,
    save_path=None
):
    removal_fractions = np.asarray(feature_cache['removal_fractions'], dtype=float)
    feat_full = feature_cache['feat_full']
    feat_occ_map = feature_cache['feat_occ_map']
    occlusion_method = feature_cache['occlusion_method']

    if verbose:
        print(f'RGE Evaluation: {model_name}')
        print(f'Occlusion: {occlusion_method}')
        print(f'Testing {len(removal_fractions)} removal fractions')
        print('Using shared cached features')

    prob_full = get_predictions_from_features(
        feat_full,
        model,
        model_class_order,
        class_order,
        model_type=model_type,
        device=device,
        batch_size=batch_size
    )

    rge_scores = []
    per_class_rge_list = []

    for frac in removal_fractions:
        if verbose:
            print(f'\nOcclusion level: {frac * 100:.0f}%')

        feat_occ = feat_occ_map[float(frac)]

        prob_occ = get_predictions_from_features(
            feat_occ,
            model,
            model_class_order,
            class_order,
            model_type=model_type,
            device=device,
            batch_size=batch_size
        )

        rge_val, rge_per_class, _ = _rge_cramer_multiclass(
            prob_full,
            prob_occ,
            class_order=class_order,
            class_weights=class_weights
        )

        rge_value = nan_to_zero(rge_val)
        rge_scores.append(rge_value)
        per_class_rge_list.append(rge_per_class)

        if verbose:
            print(f'RGE = {rge_value:.4f}')

    rge_scores = np.asarray(rge_scores, dtype=float)
    per_class_rge_list = np.asarray(per_class_rge_list, dtype=float)
    rge_rescaled = rescale_by_rga(rge_scores, rga_full)
    aurge = area_under_normalized_curve(removal_fractions, rge_rescaled)

    result = {
        'method': 'image',
        'rge_scores': rge_scores,
        'rge_rescaled': rge_rescaled,
        'aurge': aurge,
        'removal_fractions': removal_fractions,
        'per_class_rge': per_class_rge_list,
        'class_order': np.asarray(class_order),
        'occlusion_method': occlusion_method
    }

    if verbose:
        print(f'AURGE: {aurge:.4f}')

    if plot or save_path is not None:
        plot_rge(
            result,
            model_name=model_name,
            x_key='removal_fractions',
            x_label='Occluded Image Area (%)',
            x_scale=100.0,
            title=f'RGE Curve: {model_name} ({occlusion_method})',
            fig_size=fig_size,
            save_path=save_path,
            show=(plot and save_path is None)
        )

    return result


def _compare_rge_image_core(
    *,
    models,
    images_dataset,
    removal_fractions,
    class_order,
    occlusion_method='random',
    patch_size=32,
    batch_size=64,
    class_weights=None,
    rga_dict=None,
    device=None,
    verbose=True,
    random_seed=None,
    patch_rankings=None,
    patch_meta=None,
    mask_value=0.0,
    use_shared_feature_cache=True
):
    if isinstance(occlusion_method, str):
        methods = {name: occlusion_method for name in models}
    elif isinstance(occlusion_method, dict):
        methods = occlusion_method
    else:
        raise TypeError('occlusion_method must be a string or dict.')

    results = {}
    can_share_cache = use_shared_feature_cache and len(set(methods.values())) == 1

    shared_cache = None
    if can_share_cache:
        first_name = next(iter(models))
        _, preprocess_fn_first, _, _ = models[first_name]
        shared_method = methods[first_name]

        shared_cache = _precompute_rge_feature_cache(
            preprocess_fn=preprocess_fn_first,
            images_dataset=images_dataset,
            removal_fractions=removal_fractions,
            batch_size=batch_size,
            occlusion_method=shared_method,
            patch_size=patch_size,
            random_seed=random_seed,
            mask_value=mask_value,
            patch_rankings=patch_rankings,
            patch_meta=patch_meta,
            verbose=verbose
        )

    for name, (model, preprocess_fn, model_class_order, model_type) in models.items():
        if verbose:
            print(f'\nEvaluating {name}')

        rga_full = rga_dict.get(name) if rga_dict else None

        if shared_cache is not None:
            result = _rge_curve_image_cached_core(
                model=model,
                feature_cache=shared_cache,
                model_class_order=model_class_order,
                class_order=class_order,
                model_type=model_type,
                device=device,
                batch_size=batch_size,
                class_weights=class_weights,
                model_name=name,
                rga_full=rga_full,
                plot=False,
                verbose=verbose
            )
        else:
            result = _rge_curve_image_core(
                model=model,
                preprocess_fn=preprocess_fn,
                images_dataset=images_dataset,
                removal_fractions=removal_fractions,
                model_class_order=model_class_order,
                class_order=class_order,
                model_type=model_type,
                device=device,
                patch_size=patch_size,
                batch_size=batch_size,
                class_weights=class_weights,
                model_name=name,
                rga_full=rga_full,
                occlusion_method=methods.get(name, 'random'),
                patch_rankings=patch_rankings,
                patch_meta=patch_meta,
                plot=False,
                verbose=verbose,
                random_seed=random_seed,
                mask_value=mask_value
            )

        results[name] = result

    return results


# ---- Text helpers ----
def _rge_curve_text_core(
    model,
    x,
    removal_fractions,
    model_class_order,
    class_order,
    model_type='sklearn',
    device=None,
    batch_size=256,
    class_weights=None,
    model_name='Model',
    rga_full=None,
    masking_method='random',
    feature_ranking=None,
    baseline='zero',
    plot=True,
    fig_size=(10, 6),
    verbose=True,
    random_seed=None,
    save_path=None,
    prob_full_cached=None
):
    removal_fractions = np.asarray(removal_fractions, dtype=float)
    x = np.asarray(x, dtype=float)
    n_samples, n_features = x.shape

    if masking_method == 'most_important' and feature_ranking is None:
        raise ValueError("feature_ranking is required when masking_method='most_important'")

    if baseline not in ('zero', 'mean'):
        raise ValueError(f"Unknown baseline: {baseline}. Use 'zero' or 'mean'.")

    if verbose:
        print(f'RGE: {model_name}')
        print(f'Masking: {masking_method} | Baseline: {baseline}')
        print(f'x: {x.shape} | Testing {len(removal_fractions)} removal fractions')

    rng = np.random.default_rng(random_seed if random_seed is not None else 42)
    feat_mean = np.nanmean(x, axis=0) if baseline == 'mean' else None

    if prob_full_cached is None:
        prob_full = get_predictions_from_features(
            x,
            model,
            model_class_order,
            class_order,
            model_type=model_type,
            device=device,
            batch_size=batch_size
        )
    else:
        prob_full = np.asarray(prob_full_cached)

    rge_scores = []
    per_class_rge_list = []

    for frac in removal_fractions:
        frac = float(frac)

        if frac < 0 or frac > 1:
            raise ValueError(f'removal fraction must be in [0,1], got {frac}')

        k = int(np.floor(frac * n_features))

        if verbose:
            print(f'\nRemoval level: {frac * 100:.0f}% | masking {k}/{n_features} features')

        x_masked = x.copy()

        if k > 0:
            if masking_method == 'random':
                cols = rng.choice(n_features, size=k, replace=False)
            elif masking_method == 'most_important':
                cols = np.asarray(feature_ranking, dtype=int)[:k]
            else:
                raise ValueError(f'Unknown masking_method: {masking_method}')

            apply_feature_baseline(x_masked, cols, baseline=baseline, feat_mean=feat_mean)

        prob_reduced = get_predictions_from_features(
            x_masked,
            model,
            model_class_order,
            class_order,
            model_type=model_type,
            device=device,
            batch_size=batch_size
        )

        rge_val, rge_per_class, _ = _rge_cramer_multiclass(
            prob_full,
            prob_reduced,
            class_order=class_order,
            class_weights=class_weights
        )

        rge_value = nan_to_zero(rge_val)
        rge_scores.append(rge_value)
        per_class_rge_list.append(rge_per_class)

        if verbose:
            print(f'RGE = {rge_value:.4f}')

    rge_scores = np.asarray(rge_scores, dtype=float)
    per_class_rge_list = np.asarray(per_class_rge_list, dtype=float)
    rge_rescaled = rescale_by_rga(rge_scores, rga_full)
    aurge = area_under_normalized_curve(removal_fractions, rge_rescaled)

    result = {
        'method': 'text',
        'rge_scores': rge_scores,
        'rge_rescaled': rge_rescaled,
        'aurge': aurge,
        'removal_fractions': removal_fractions,
        'per_class_rge': per_class_rge_list,
        'class_order': np.asarray(class_order),
        'masking_method': masking_method,
        'baseline': baseline
    }

    if verbose:
        print(f'AURGE: {aurge:.4f}')

    if plot or save_path is not None:
        plot_rge(
            result,
            model_name=model_name,
            x_key='removal_fractions',
            x_label='Removed Features (%)',
            x_scale=100.0,
            title=f'RGE Curve: {model_name} ({masking_method})',
            fig_size=fig_size,
            save_path=save_path,
            show=(plot and save_path is None)
        )

    return result


def _compare_rge_text_core(
    *,
    models,
    removal_fractions,
    class_order,
    masking_method='random',
    baseline='zero',
    class_weights=None,
    rga_dict=None,
    batch_size=256,
    verbose=True,
    random_seed=None,
    feature_rankings=None
):
    results = {}
    rankings_map = normalize_rankings(feature_rankings, models)

    for name, tpl in models.items():
        if len(tpl) != 6:
            raise ValueError(
                f"models['{name}'] must be "
                "(model, x, prob_full, model_class_order, model_type, device)."
            )

        model, x, prob_full, model_class_order, model_type, device = tpl

        if verbose:
            print(f'\nEvaluating RGE for {name}')

        results[name] = _rge_curve_text_core(
            model=model,
            x=x,
            removal_fractions=removal_fractions,
            model_class_order=model_class_order,
            class_order=class_order,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            class_weights=class_weights,
            model_name=name,
            rga_full=(rga_dict.get(name) if rga_dict else None),
            masking_method=masking_method,
            feature_ranking=rankings_map.get(name),
            baseline=baseline,
            plot=False,
            verbose=verbose,
            random_seed=random_seed,
            prob_full_cached=prob_full
        )

    return results


# ---- Tabular helpers ----
def _rge_curve_tabular_core(
    model,
    x,
    feature_names,
    model_class_order,
    class_order,
    model_type='sklearn',
    device=None,
    class_weights=None,
    model_name='Model',
    rga_full=None,
    masking_method='greedy',
    feature_ranking=None,
    baseline='zero',
    n_steps=None,
    random_seed=None,
    verbose=True,
    plot=False,
    fig_size=(10, 6),
    save_path=None,
    prob_full_cached=None
):
    x = np.asarray(x, dtype=float)
    feature_names = list(feature_names)
    n_samples, n_features = x.shape

    if n_steps is None:
        n_steps = n_features

    if baseline not in ('zero', 'mean'):
        raise ValueError(f"Unknown baseline: {baseline}. Use 'zero' or 'mean'.")

    rng = np.random.default_rng(random_seed if random_seed is not None else 42)
    feat_mean = np.nanmean(x, axis=0) if baseline == 'mean' else None

    if prob_full_cached is None:
        prob_full = get_predictions_from_features(
            x,
            model,
            model_class_order,
            class_order,
            model_type=model_type,
            device=device,
            batch_size=256
        )
    else:
        prob_full = np.asarray(prob_full_cached)

    removed = []
    remaining = list(range(n_features))
    rge_scores = [1.0]
    per_class_rge_list = []

    for step in range(1, n_steps + 1):
        if verbose:
            print(f'[RGE-tabular] step {step}/{n_steps} | removed={len(removed)}')

        if masking_method == 'random':
            k = min(step, n_features)
            cols = rng.choice(n_features, size=k, replace=False)
            x_masked = mask_columns(x, cols, baseline=baseline, feat_mean=feat_mean)

            prob_reduced = get_predictions_from_features(
                x_masked,
                model,
                model_class_order,
                class_order,
                model_type=model_type,
                device=device,
                batch_size=256
            )

            rge_val, rge_per_class, _ = _rge_cramer_multiclass(
                prob_full,
                prob_reduced,
                class_order=class_order,
                class_weights=class_weights
            )
            rge_scores.append(nan_to_zero(rge_val))
            per_class_rge_list.append(rge_per_class)

        elif masking_method == 'most_important':
            if feature_ranking is None:
                raise ValueError("feature_ranking required for masking_method='most_important'")

            cols = np.asarray(feature_ranking, dtype=int)[:step]
            x_masked = mask_columns(x, cols, baseline=baseline, feat_mean=feat_mean)

            prob_reduced = get_predictions_from_features(
                x_masked,
                model,
                model_class_order,
                class_order,
                model_type=model_type,
                device=device,
                batch_size=256
            )

            rge_val, rge_per_class, _ = _rge_cramer_multiclass(
                prob_full,
                prob_reduced,
                class_order=class_order,
                class_weights=class_weights
            )
            rge_scores.append(nan_to_zero(rge_val))
            per_class_rge_list.append(rge_per_class)

        elif masking_method == 'greedy':
            best_j = None
            best_rge = -np.inf
            best_per_class = None

            for j in remaining:
                cols = removed + [j]
                x_masked = mask_columns(x, cols, baseline=baseline, feat_mean=feat_mean)

                prob_reduced = get_predictions_from_features(
                    x_masked,
                    model,
                    model_class_order,
                    class_order,
                    model_type=model_type,
                    device=device,
                    batch_size=256
                )

                rge_val, rge_per_class, _ = _rge_cramer_multiclass(
                    prob_full,
                    prob_reduced,
                    class_order=class_order,
                    class_weights=class_weights
                )

                candidate = -np.inf if np.isnan(rge_val) else float(rge_val)
                if candidate > best_rge:
                    best_rge = candidate
                    best_j = j
                    best_per_class = rge_per_class

            removed.append(best_j)
            remaining.remove(best_j)

            rge_scores.append(0.0 if not np.isfinite(best_rge) else float(best_rge))
            per_class_rge_list.append(best_per_class)

            if verbose:
                print(f'picked: {feature_names[best_j]} | rge={rge_scores[-1]:.4f}')

        else:
            raise ValueError(f'Unknown masking_method: {masking_method}')

    rge_scores = np.asarray(rge_scores, dtype=float)
    x_axis = np.linspace(0, 1, len(rge_scores))
    rge_rescaled = rescale_by_rga(rge_scores, rga_full)
    aurge = area_under_normalized_curve(x_axis, rge_rescaled)
    removal_fractions = np.linspace(0, 1, n_steps + 1)

    result = {
        'method': 'tabular',
        'x_axis': x_axis,
        'removal_fractions': removal_fractions,
        'rge_scores': rge_scores,
        'rge_rescaled': rge_rescaled,
        'aurge': aurge,
        'removed_features': [feature_names[i] for i in removed] if masking_method == 'greedy' else None,
        'per_class_rge': np.asarray(per_class_rge_list, dtype=float) if len(per_class_rge_list) else None,
        'class_order': np.asarray(class_order),
        'masking_method': masking_method,
        'baseline': baseline
    }

    if verbose:
        print(f'AURGE: {aurge:.4f}')

    if plot or save_path is not None:
        plot_rge(
            result,
            model_name=model_name,
            x_key='x_axis',
            x_label='Fraction of Features Removed',
            x_scale=1.0,
            title=f'RGE Curve: {model_name}',
            fig_size=fig_size,
            save_path=save_path,
            show=(plot and save_path is None)
        )

    return result


def _compare_rge_tabular_core(
    *,
    models,
    class_order,
    class_weights=None,
    rga_dict=None,
    masking_method='greedy',
    baseline='zero',
    n_steps=None,
    verbose=True,
    random_seed=None,
    feature_rankings=None
):
    results = {}
    rankings_map = normalize_rankings(feature_rankings, models)

    for name, tpl in models.items():
        if len(tpl) != 7:
            raise ValueError(
                f"models['{name}'] must be "
                "(model, x, feature_names, prob_full, model_class_order, model_type, device)."
            )

        model, x, feature_names, prob_full, model_class_order, model_type, device = tpl

        if verbose:
            print(f'\nEvaluating RGE (tabular) for {name}')

        results[name] = _rge_curve_tabular_core(
            model=model,
            x=x,
            feature_names=feature_names,
            model_class_order=model_class_order,
            class_order=class_order,
            model_type=model_type,
            device=device,
            class_weights=class_weights,
            model_name=name,
            rga_full=(rga_dict.get(name) if rga_dict else None),
            masking_method=masking_method,
            feature_ranking=rankings_map.get(name),
            baseline=baseline,
            n_steps=n_steps,
            random_seed=random_seed,
            verbose=verbose,
            prob_full_cached=prob_full
        )

    return results


# ---- Private helpers ----
def _is_single_rge_result(result):
    return isinstance(result, dict) and 'aurge' in result and (
        'removal_fractions' in result or 'x_axis' in result
    )


def _infer_x_key(result):
    if 'x_axis' in result:
        return 'x_axis'
    if 'removal_fractions' in result:
        return 'removal_fractions'
    raise ValueError('Cannot infer x-axis key from result.')


def _default_x_label(x_key):
    if x_key == 'x_axis':
        return 'Fraction of Features Removed'
    if x_key == 'removal_fractions':
        return 'Removal Fraction'
    return 'Fraction'


def _curve_label(model_name, result):
    aurge = result.get('aurge', np.nan)
    method = result.get('method', None)

    if np.isfinite(aurge):
        if method is not None:
            return f'{model_name} (AURGE={aurge:.3f}, {method})'
        return f'{model_name} (AURGE={aurge:.3f})'

    return f'{model_name}'


def _print_comparison_summary(results, *, metric_name='AURGE'):
    print('Explainability Comparison Summary')

    names = list(results.keys())
    scores = np.asarray([results[name].get('aurge', np.nan) for name in names], dtype=float)

    for name, score in zip(names, scores):
        print(f'{name}: {metric_name} = {score:.4f}')

    finite = np.isfinite(scores)
    if len(names) >= 2 and np.any(finite):
        finite_idx = np.where(finite)[0]
        best_idx = finite_idx[int(np.argmax(scores[finite]))]
        worst_idx = finite_idx[int(np.argmin(scores[finite]))]

        print(f'Best: {names[best_idx]} ({metric_name}={scores[best_idx]:.4f})')
        print(f'Worst: {names[worst_idx]} ({metric_name}={scores[worst_idx]:.4f})')
