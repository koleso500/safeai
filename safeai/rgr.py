"""
Rank Graduation Robustness (RGR).

Main functions
--------------
rgr_score
    Compute one RGR value from original and perturbed predictions.

rgr_curve
    Compute one RGR robustness curve for one model.

aurgr_score
    Compute only the area under an RGR curve.

compare_rgr
    Compare several models using one of the supported RGR workflows.

plot_rgr
    Plot one RGR curve or a comparison of several RGR curves.

Notes
-----
RGR measures stability of model predictions under perturbations. Values closer
to 1 indicate stronger robustness, while values closer to 0 indicate stronger
prediction degradation.

The recommended user-facing function is compare_rgr(..., method=...). It can
run Gaussian-noise robustness, adversarial robustness, Wasserstein image-level
robustness, or spatial image-level robustness from the same entry point.

"""

from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn

from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    SquareAttack,
    HopSkipJump,
    SimBA,
    Wasserstein,
    SpatialTransformation
)
from art.estimators.classification import PyTorchClassifier, SklearnClassifier

from safeai.cramer import gini_via_lorenz, cvm1_concordance_weighted
from safeai.utils import (
    align_proba_to_class_order,
    ensure_prob_matrix,
    clean_pair,
    validate_method,
    validate_class_weights,
    rescale_by_rga,
    area_under_normalized_curve,
    nan_to_zero,
    resolve_class_orders
)


AttackName = Literal['fgsm', 'pgd', 'square', 'hsj', 'simba']
RGRMethod = Literal['noise', 'adversarial', 'wasserstein_images', 'spatial_images']
FeatureRGRMethod = Literal['noise', 'adversarial']


__all__ = [
    'rgr_score',
    'rgr_curve',
    'aurgr_score',
    'compare_rgr',
    'plot_rgr'
]


# ---- Public API ----
def rgr_score(
    pred_original,
    pred_perturbed,
    *,
    class_order=None,
    class_weights=None,
    verbose=False
):
    """
    Compute Rank Graduation Robustness between two prediction arrays.

    Parameters
    ----------
    pred_original : array-like
        Original predictions. Can be a 1D score vector or a 2D probability
        matrix.

    pred_perturbed : array-like
        Predictions after perturbation. Must have the same shape as
        pred_original.

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
        RGR score.
    """
    pred_original = np.asarray(pred_original)
    pred_perturbed = np.asarray(pred_perturbed)

    if pred_original.ndim == 1 and pred_perturbed.ndim == 1:
        if class_order is None:
            return _rgr_cramer(pred_original, pred_perturbed)

        score, _, _ = _rgr_cramer_multiclass(
            pred_original,
            pred_perturbed,
            class_order=class_order,
            class_weights=class_weights,
            verbose=verbose
        )
        return score

    score, _, _ = _rgr_cramer_multiclass(
        pred_original,
        pred_perturbed,
        class_order=class_order,
        class_weights=class_weights,
        verbose=verbose
    )
    return score


def rgr_curve(
    model,
    x_data,
    strengths,
    *,
    method: FeatureRGRMethod = 'noise',
    prob_original=None,
    model_class_order=None,
    class_order=None,
    y_true=None,
    attack_name: AttackName = 'fgsm',
    base_attack_params=None,
    class_weights=None,
    model_type='sklearn',
    device=None,
    rga_full=None,
    model_name='Model',
    random_seed=None,
    plot=False,
    fig_size=(10, 6),
    save_path=None,
    show=False,
    verbose=True
):
    """
    Compute one Rank Graduation Robustness (RGR) curve and its AURGR value.

    This function evaluates the robustness of one trained model under increasing
    feature-space perturbation strength. It supports two single-model workflows:

    - Gaussian noise perturbation, using method='noise'
    - Adversarial perturbation, using method='adversarial'

    Image-level robustness workflows, such as Wasserstein and spatial
    transformations, are handled by compare_rgr(...), because they require a shared
    attack model and preprocessing pipeline across several models.

    Parameters
    ----------
    model : object
        Trained model to evaluate.

        For model_type='sklearn', the model must provide predict_proba(...).

        For model_type='pytorch', the model must return logits when called on an
        input tensor.

    x_data : array-like or torch.Tensor
        Input feature matrix used for robustness evaluation. For sklearn models,
        this is usually a NumPy array or pandas-compatible matrix. For PyTorch
        models, this may be either a tensor or an array that can be converted to a
        float tensor.

    strengths : array-like
        Perturbation strengths used to build the RGR curve.

        For method='noise', values are interpreted as Gaussian noise standard
        deviations.

        For method='adversarial', values are interpreted according to the selected
        ART attack. For FGSM, PGD, and Square Attack, these are eps values. For
        SimBA, these are epsilon values. For HopSkipJump, these are interpreted as
        max_iter values.

    method : {'noise', 'adversarial'}, default='noise'
        Feature-space robustness workflow used to construct the curve.

    prob_original : array-like, optional
        Original predicted probabilities for x_data.

        If None, probabilities are computed from model and x_data. If provided,
        the probabilities should follow model_class_order and will be aligned to
        class_order.

    model_class_order : array-like, optional
        Class order produced by the model probability output.

        For sklearn models, this is usually ``model.classes_``. If omitted, the
        function tries to infer it from the model or from prob_original.

    class_order : array-like, optional
        Target class order used to align probability columns across models.

        If omitted, model_class_order is used.

    y_true : array-like, optional
        True class labels. Required when method='adversarial', because ART attacks
        need labels to generate adversarial examples.

    attack_name : {'fgsm', 'pgd', 'square', 'hsj', 'simba'}, default='fgsm'
        ART adversarial attack used when method='adversarial'.

    base_attack_params : dict, optional
        Additional fixed parameters passed to the ART attack constructor. The
        current strength value is inserted automatically using the appropriate
        parameter name for the selected attack.

    class_weights : array-like, optional
        Weights used to aggregate per-class RGR values. If None, uniform class
        weights are used.

    model_type : {'sklearn', 'pytorch'}, default='sklearn'
        Type of model being evaluated.

    device : torch.device or str, optional
        Device used for PyTorch inference. Ignored for sklearn models.

    rga_full : float, optional
        Full RGA score used to rescale the RGR curve. If provided and finite,
        rgr_rescaled = rgr_scores * rga_full. If None, no rescaling is applied.

    model_name : str, default='Model'
        Name stored in the result dictionary and used in plot labels.

    random_seed : int, optional
        Random seed used for Gaussian noise generation when method='noise'.

    plot : bool, default=False
        Whether to create a plot for the computed RGR curve.

    fig_size : tuple, default=(10, 6)
        Figure size used when plotting.

    save_path : str, optional
        Path where the plot should be saved. If provided, the plot is written to
        disk.

    show : bool, default=False
        Whether to display the plot with plt.show().

    verbose : bool, default=True
        Whether to print progress and summary information.

    Returns
    -------
    dict
        Dictionary with the computed RGR curve and metadata.

        Common keys include:

        - 'rgr_scores' : np.ndarray
            Raw RGR scores at each perturbation strength.
        - 'rgr_rescaled' : np.ndarray
            RGR scores after optional rescaling by rga_full.
        - 'aurgr' : float
            Area under the RGR curve.
        - 'per_class_rgr' : np.ndarray
            Per-class RGR values at each perturbation strength.
        - 'class_order' : np.ndarray
            Class order used for probability alignment.
        - 'method' : str
            Robustness workflow used to build the curve.
        - 'model_name' : str
            Name of the evaluated model.

        For method='noise', the result also contains:

        - 'noise_levels' : np.ndarray

        For method='adversarial', the result also contains:

        - 'attack_name' : str
        - 'attack_strengths' : np.ndarray
    """
    validate_method(method, allowed={'noise', 'adversarial'})

    model_class_order, class_order = resolve_class_orders(
        model,
        model_class_order=model_class_order,
        class_order=class_order,
        prob=prob_original
    )

    if prob_original is None:
        prob_original = _predict_probabilities(
            model,
            x_data,
            model_type=model_type,
            device=device
        )

    if method == 'noise':
        result = _rgr_curve_noise_core(
            model=model,
            x_data=x_data,
            prob_original=prob_original,
            noise_levels=strengths,
            model_class_order=model_class_order,
            class_order=class_order,
            class_weights=class_weights,
            model_type=model_type,
            device=device,
            rga_full=rga_full,
            model_name=model_name,
            plot=False,
            fig_size=fig_size,
            verbose=verbose,
            random_seed=random_seed,
            save_path=None
        )
    else:
        if y_true is None:
            raise ValueError("y_true is required when method='adversarial'.")

        result = _rgr_curve_adversarial_core(
            model=model,
            x_data=x_data,
            prob_original=prob_original,
            attack_strengths=strengths,
            model_class_order=model_class_order,
            class_order=class_order,
            y_true=y_true,
            attack_name=attack_name,
            base_attack_params=base_attack_params,
            class_weights=class_weights,
            model_type=model_type,
            device=device,
            rga_full=rga_full,
            model_name=model_name,
            plot=False,
            fig_size=fig_size,
            verbose=verbose,
            save_path=None
        )

    result['method'] = method
    result['model_name'] = model_name

    if plot or save_path is not None or show:
        plot_rgr(result, model_name=model_name, fig_size=fig_size, save_path=save_path, show=show)

    return result


def aurgr_score(
    model,
    x_data,
    strengths,
    **kwargs
):
    """
    Compute only the area under an RGR curve.
    """
    result = rgr_curve(model, x_data, strengths, plot=False, **kwargs)
    return result['aurgr']


def compare_rgr(
    models,
    strengths,
    class_order,
    *,
    method: RGRMethod = 'noise',
    y_true=None,
    y_true_dict=None,
    images=None,
    attack_model=None,
    preprocess_fn=None,
    attack_name: AttackName = 'fgsm',
    base_attack_params=None,
    rga_dict=None,
    class_weights=None,
    fig_size=(12, 6),
    verbose=True,
    random_seed=None,
    save_path=None,
    show=False,
    max_iter=50,
    eps_step=0.01,
    num_translations=3,
    num_rotations=3
):
    """
    Compare several models using Rank Graduation Robustness (RGR) curves.

    This is the main user-facing comparison function for RGR. It provides one
    unified interface for all supported robustness workflows:

    - Gaussian feature noise, using method='noise'
    - Feature-space adversarial attacks, using method='adversarial'
    - Image-level Wasserstein attacks, using method='wasserstein_images'
    - Image-level spatial transformations, using method='spatial_images'

    The function returns one result dictionary per model and can optionally plot a
    comparison of the resulting RGR curves.

    Parameters
    ----------
    models : dict
        Dictionary containing model configurations.

        For method='noise' or method='adversarial', each entry must have the form::

            model_name -> (
                model,
                x_data,
                prob_original,
                model_class_order,
                model_type,
                device
            )

        where:

        - model is a trained sklearn estimator or PyTorch module.
        - x_data is the feature matrix used for perturbation.
        - prob_original is the original probability matrix, or None.
        - model_class_order is the class order of the model probability output.
        - model_type is either 'sklearn' or 'pytorch'.
        - device is the torch device for PyTorch models, or None for sklearn.

        For method='wasserstein_images' or method='spatial_images', each entry
        must have the form::

            model_name -> (
                model,
                prob_original,
                model_class_order,
                model_type,
                device
            )

        In image-level workflows, perturbed images are first generated using
        attack_model, then converted to model-ready features through preprocess_fn.

    strengths : array-like
        Perturbation strengths used to build the RGR curves.

        Their meaning depends on method:

        - method='noise':
            Gaussian noise standard deviations.

        - method='adversarial':
            Attack strengths for the selected ART attack.

        - method='wasserstein_images':
            Wasserstein attack eps values.

        - method='spatial_images':
            Spatial transformation strengths.

    class_order : array-like
        Shared target class order used to align probability columns across all
        models.

    method : {'noise', 'adversarial', 'wasserstein_images', 'spatial_images'}, default='noise'
        Robustness workflow used for all models in the comparison.

    y_true : array-like, optional
        Shared true labels.

        Required for method='wasserstein_images' and method='spatial_images'.

        For method='adversarial', either y_true or y_true_dict must be provided.

    y_true_dict : dict, optional
        Per-model true labels for adversarial evaluation. This is useful when
        different models are evaluated on different input arrays.

    images : array-like or torch.Tensor, optional
        Original image tensor or image array used for image-level robustness
        evaluation. Required for method='wasserstein_images' and
        method='spatial_images'.

    attack_model : torch.nn.Module, optional
        PyTorch model used by ART to generate image-level adversarial examples or
        spatial transformations. Required for image-level methods.

    preprocess_fn : callable, optional
        Function that maps perturbed images to model-ready feature matrices.
        Required for image-level methods.

    attack_name : {'fgsm', 'pgd', 'square', 'hsj', 'simba'}, default='fgsm'
        ART adversarial attack used when method='adversarial'.

    base_attack_params : dict, optional
        Additional fixed parameters passed to the ART attack constructor. The
        current strength value is inserted automatically using the appropriate
        parameter name for the selected attack.

    rga_dict : dict, optional
        Mapping from model name to full RGA score. If provided, each RGR
        curve is rescaled by the corresponding RGA value.

    class_weights : array-like, optional
        Weights used to aggregate per-class RGR values. If None, uniform class
        weights are used.

    fig_size : tuple, default=(12, 6)
        Figure size used for the comparison plot.

    verbose : bool, default=True
        Whether to print progress and summary information.

    random_seed : int, optional
        Random seed used for Gaussian noise generation when method='noise'.

    save_path : str, optional
        Path where the comparison plot should be saved. If None and show=False,
        no plot is saved.

    show : bool, default=False
        Whether to display the comparison plot with plt.show().

    max_iter : int, default=50
        Maximum number of iterations used by the Wasserstein image attack.

    eps_step : float, default=0.01
        Step size used by the Wasserstein image attack.

    num_translations : int, default=3
        Number of translations tested by the spatial transformation attack.

    num_rotations : int, default=3
        Number of rotations tested by the spatial transformation attack.

    Returns
    -------
    dict
        Mapping from model name to RGR result dictionary.

        Each result contains:

        - 'rgr_scores' : np.ndarray or list
            Raw RGR scores at each perturbation strength.
        - 'rgr_rescaled' : np.ndarray or list
            RGR scores after optional rescaling by rga_dict.
        - 'aurgr' : float
            Area under the RGR curve.
        - 'per_class_rgr' : np.ndarray or list
            Per-class RGR values at each perturbation strength.
        - 'class_order' : np.ndarray or list
            Class order used for probability alignment.
        - 'method' : str
            Robustness workflow used in the comparison.

        Depending on method, each result also includes one of:

        - 'noise_levels'
        - 'attack_strengths'

        For adversarial methods, results also include:

        - 'attack_name'
    """
    validate_method(
        method,
        allowed={'noise', 'adversarial', 'wasserstein_images', 'spatial_images'}
    )

    if method == 'noise':
        results = _compare_rgr_noise_core(
            models=models,
            noise_levels=strengths,
            class_order=class_order,
            rga_dict=rga_dict,
            class_weights=class_weights,
            verbose=verbose,
            random_seed=random_seed
        )
        x_key = 'noise_levels'
        x_label = 'Noise Standard Deviation'
        title = 'RGR Curves Comparison'

    elif method == 'adversarial':
        results = _compare_rgr_adversarial_core(
            models=models,
            attack_strengths=strengths,
            class_order=class_order,
            y_true=y_true,
            y_true_dict=y_true_dict,
            attack_name=attack_name,
            rga_dict=rga_dict,
            class_weights=class_weights,
            verbose=verbose,
            base_attack_params=base_attack_params
        )
        x_key = 'attack_strengths'
        x_label = 'Attack strength ε'
        title = f'Adversarial RGR Curves Comparison ({attack_name.upper()})'

    elif method == 'wasserstein_images':
        if images is None or y_true is None or attack_model is None or preprocess_fn is None:
            raise ValueError(
                "images, y_true, attack_model, and preprocess_fn are required "
                "when method='wasserstein_images'."
            )

        results = _compare_rgr_wasserstein_images_core(
            models=models,
            images=images,
            y_true=y_true,
            attack_model=attack_model,
            preprocess_fn=preprocess_fn,
            attack_strengths=strengths,
            class_order=class_order,
            rga_dict=rga_dict,
            class_weights=class_weights,
            verbose=verbose,
            max_iter=max_iter,
            eps_step=eps_step
        )
        x_key = 'attack_strengths'
        x_label = 'Wasserstein attack strength ε'
        title = 'Image-level Wasserstein RGR Curves'

    else:  # spatial_images
        if images is None or y_true is None or attack_model is None or preprocess_fn is None:
            raise ValueError(
                "images, y_true, attack_model, and preprocess_fn are required "
                "when method='spatial_images'."
            )

        results = _compare_rgr_spatial_images_core(
            models=models,
            images=images,
            y_true=y_true,
            attack_model=attack_model,
            preprocess_fn=preprocess_fn,
            attack_strengths=strengths,
            class_order=class_order,
            rga_dict=rga_dict,
            class_weights=class_weights,
            verbose=verbose,
            num_translations=num_translations,
            num_rotations=num_rotations
        )
        x_key = 'attack_strengths'
        x_label = 'Spatial attack strength'
        title = 'Image-level Spatial Transformation RGR Curves'

    results = cast(dict[str, dict[str, Any]], results)

    for result in results.values():
        result['method'] = method

    if save_path is not None or show:
        plot_rgr(
            results,
            x_key=x_key,
            x_label=x_label,
            title=title,
            fig_size=fig_size,
            save_path=save_path,
            show=show
        )

    if verbose:
        _print_comparison_summary(results, metric_name='AURGR')

    return results


def plot_rgr(
    result,
    *,
    model_name='Model',
    x_key=None,
    x_label=None,
    y_key='rgr_rescaled',
    title=None,
    fig_size=(12, 6),
    save_path=None,
    show=False
):
    """
    Plot one RGR curve or a comparison of several RGR curves.

    Parameters
    ----------
    result : dict
        Either one result returned by rgr_curve(...) or the full dictionary
        returned by compare_rgr(...).

    model_name : str, default='Model'
        Label used when result is a single curve.

    x_key : str, optional
        Name of the x-axis field in each result. If None, it is inferred from
        the result keys, for example 'noise_levels' or 'attack_strengths'.

    x_label : str, optional
        Human-readable x-axis label. If None, a default label is selected.

    y_key : str, default='rgr_rescaled'
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

    if _is_single_rgr_result(result):
        results = {model_name: result}
    else:
        results = result

    if x_key is None:
        first = next(iter(results.values()))
        x_key = _infer_x_key(first)

    if x_label is None:
        x_label = _default_x_label(x_key)

    if title is None:
        title = 'RGR Curve' if len(results) == 1 else 'RGR Curves Comparison'

    fig, ax = plt.subplots(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (name, res), color in zip(results.items(), colors):
        x_values = np.asarray(res[x_key], dtype=float)
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
    ax.set_ylabel('RGR Score', fontsize=11, fontweight='bold')
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


# ---- Scalar RGR helpers ----
def _rgr_cramer(pred, pred_pert):
    """
    RGR between original predictions and perturbed predictions.
    """
    pred, pred_pert = clean_pair(pred, pred_pert)

    if len(pred) == 0:
        return np.nan

    g = gini_via_lorenz(pred)
    if not np.isfinite(g) or g == 0:
        return np.nan

    cvm = cvm1_concordance_weighted(pred, pred_pert)
    if not np.isfinite(cvm):
        return np.nan

    return float(1 - cvm / g)


def _rgr_cramer_multiclass(
    prob_original,
    prob_perturbed,
    class_order=None,
    class_weights=None,
    verbose=False
):
    """
    Calculate weighted one-vs-rest RGR for multiclass classification.
    """
    prob_original = np.asarray(prob_original, dtype=float)
    prob_perturbed = np.asarray(prob_perturbed, dtype=float)

    if class_order is not None:
        class_order = np.asarray(class_order)
        prob_original = ensure_prob_matrix(prob_original, class_order)
        prob_perturbed = ensure_prob_matrix(prob_perturbed, class_order)
    else:
        if prob_original.ndim != 2 or prob_perturbed.ndim != 2:
            raise ValueError('For 1D binary probabilities, pass class_order with 2 classes.')

    if prob_original.shape != prob_perturbed.shape:
        raise ValueError(
            f'Shape mismatch: prob_original {prob_original.shape} and '
            f'prob_perturbed {prob_perturbed.shape}.'
        )

    if prob_original.ndim != 2:
        raise ValueError('prob_original and prob_perturbed must be 2D after conversion.')

    n_classes = prob_original.shape[1]
    class_weights = validate_class_weights(class_weights, n_classes)

    rgrs = []

    for k in range(n_classes):
        rgr_k = _rgr_cramer(prob_original[:, k], prob_perturbed[:, k])
        rgrs.append(rgr_k)

        if verbose:
            print(f'Class {k}: RGR = {rgr_k:.4f}')

    rgrs = np.asarray(rgrs, dtype=float)
    valid = np.isfinite(rgrs) & np.isfinite(class_weights) & (class_weights > 0)

    if np.any(valid):
        rgr_weighted = np.sum(rgrs[valid] * class_weights[valid]) / np.sum(class_weights[valid])
    else:
        rgr_weighted = np.nan

    return float(rgr_weighted) if np.isfinite(rgr_weighted) else np.nan, rgrs, class_weights


# ---- Gaussian-noise curve helper ----
def _rgr_curve_noise_core(
    model,
    x_data,
    prob_original,
    noise_levels,
    model_class_order,
    class_order,
    class_weights=None,
    model_type='sklearn',
    device=None,
    rga_full=None,
    model_name='Model',
    plot=True,
    fig_size=(10, 6),
    verbose=True,
    random_seed=None,
    save_path=None
):
    """
    Evaluate RGR robustness with Gaussian noise perturbation.
    """
    noise_levels = np.asarray(noise_levels, dtype=float)
    model_class_order = np.asarray(model_class_order)
    class_order = np.asarray(class_order)

    prob_original_aligned = align_proba_to_class_order(
        prob_original,
        model_class_order,
        class_order
    )

    rng = np.random.default_rng(random_seed)
    x_input = _prepare_x_for_model(x_data, model_type=model_type)

    rgr_scores = []
    per_class_rgr_list = []

    if verbose:
        print(f'RGR Evaluation: {model_name}')
        print(f'Testing {len(noise_levels)} noise levels')

    for i in range(len(noise_levels)):
        sigma = float(noise_levels[i])
        x_noisy = _add_gaussian_noise(x_input, sigma, model_type=model_type, rng=rng)

        prob_perturbed_raw = _predict_probabilities(
            model,
            x_noisy,
            model_type=model_type,
            device=device
        )
        prob_perturbed = align_proba_to_class_order(
            prob_perturbed_raw,
            model_class_order,
            class_order
        )

        rgr_val, rgr_per_class, _ = _rgr_cramer_multiclass(
            prob_original_aligned,
            prob_perturbed,
            class_order=class_order,
            class_weights=class_weights,
            verbose=False
        )

        rgr_scores.append(nan_to_zero(rgr_val))
        per_class_rgr_list.append(rgr_per_class)

        if verbose and i % max(1, len(noise_levels) // 5) == 0:
            print(f'σ = {sigma:.3f}: RGR = {rgr_scores[-1]:.4f}')

    rgr_scores = np.asarray(rgr_scores, dtype=float)
    per_class_rgr_list = np.asarray(per_class_rgr_list, dtype=float)
    rgr_rescaled = rescale_by_rga(rgr_scores, rga_full)
    aurgr = area_under_normalized_curve(noise_levels, rgr_rescaled)

    result = {
        'method': 'noise',
        'rgr_scores': rgr_scores,
        'rgr_rescaled': rgr_rescaled,
        'aurgr': aurgr,
        'noise_levels': noise_levels,
        'strengths': noise_levels,
        'per_class_rgr': per_class_rgr_list,
        'class_order': class_order
    }

    if verbose:
        print(f'AURGR: {aurgr:.4f}')

    if plot or save_path is not None:
        plot_rgr(
            result,
            model_name=model_name,
            x_key='noise_levels',
            x_label='Noise Standard Deviation',
            title=f'RGR Curve: {model_name}',
            fig_size=fig_size,
            save_path=save_path,
            show=(plot and save_path is None)
        )

    return result


# ---- Core comparison helpers ----
def _compare_rgr_noise_core(
    *,
    models,
    noise_levels,
    class_order,
    rga_dict=None,
    class_weights=None,
    verbose=True,
    random_seed=None
):
    results = {}

    for model_name, model_config in models.items():
        model, x_data, prob_original, model_class_order, model_type, device = model_config
        rga_full = rga_dict.get(model_name) if rga_dict else None

        if verbose:
            print(f'\nEvaluating {model_name}...')

        results[model_name] = _rgr_curve_noise_core(
            model=model,
            x_data=x_data,
            prob_original=prob_original,
            noise_levels=noise_levels,
            model_class_order=model_class_order,
            class_order=class_order,
            class_weights=class_weights,
            model_type=model_type,
            device=device,
            rga_full=rga_full,
            model_name=model_name,
            plot=False,
            verbose=verbose,
            random_seed=random_seed
        )

    return results


def _compare_rgr_adversarial_core(
    *,
    models,
    attack_strengths,
    class_order,
    y_true=None,
    y_true_dict=None,
    attack_name: AttackName = 'fgsm',
    rga_dict=None,
    class_weights=None,
    verbose=True,
    base_attack_params=None
):
    results = {}

    for model_name, model_config in models.items():
        model, x_data, prob_original, model_class_order, model_type, device = model_config

        labels = _resolve_y_true_for_model(model_name, y_true=y_true, y_true_dict=y_true_dict)
        rga_full = rga_dict.get(model_name) if rga_dict else None

        if verbose:
            print(f'\nEvaluating {model_name} with {attack_name.upper()}...')

        results[model_name] = _rgr_curve_adversarial_core(
            model=model,
            x_data=x_data,
            prob_original=prob_original,
            attack_strengths=attack_strengths,
            model_class_order=model_class_order,
            class_order=class_order,
            y_true=labels,
            attack_name=attack_name,
            base_attack_params=base_attack_params,
            class_weights=class_weights,
            model_type=model_type,
            device=device,
            rga_full=rga_full,
            model_name=model_name,
            plot=False,
            verbose=verbose
        )

    return results


def _bound_values(x_data):
    """
    Estimate the minimum and maximum values in the dataset for ART
    """
    x_np = x_data.detach().cpu().numpy() if isinstance(x_data, torch.Tensor) else np.asarray(x_data)
    x_min = float(np.min(x_np))
    x_max = float(np.max(x_np))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError('Invalid values in x_data')
    if x_min == x_max:
        x_max = x_min + 1e-6
    bounds = (x_min, x_max)
    return bounds


def _art_classifier(
    model,
    x_data,
    nb_classes,
    model_type='sklearn',
    device=None,
    clip_values=None
):
    """
    ART classifier
    """
    if clip_values is None:
        clip_values = _bound_values(x_data)

    if model_type == 'sklearn':
        return SklearnClassifier(model=model, clip_values=clip_values)

    if model_type == 'pytorch':
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data, dtype=torch.float32)

        input_shape = tuple(x_data.shape[1:])
        loss = nn.CrossEntropyLoss()

        device_type = 'gpu' if (
            device is not None and str(device).startswith('cuda') and torch.cuda.is_available()
        ) else 'cpu'

        return PyTorchClassifier(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=None,
            clip_values=clip_values,
            device_type=device_type
        )

    raise ValueError(f'Unsupported model_type: {model_type}')


def _generate_adversarial_examples(
    model,
    x_data,
    y_labels,
    nb_classes,
    attack_name: AttackName = 'fgsm',
    attack_params=None,
    model_type='sklearn',
    device=None,
    clip_values=None
):
    """
    Generate adversarial examples with ART

    Parameters
    ----------
    model :
        Trained model
    x_data :
        Input features
    y_labels :
        Integer class labels
    nb_classes : int
        Number of classes
    attack_name : {'fgsm', 'pgd', 'square', 'hsj', 'simba'}
        Attack type
    attack_params : dict, optional
        ART attack parameters
    model_type : {'sklearn', 'pytorch'}
        Model type
    device :
        Torch device for pytorch models
    clip_values : tuple, optional
        (min, max) bounds for ART

    Returns
    -------
    np.ndarray
        Adversarial samples
    """
    if attack_params is None:
        attack_params = {}

    x_np = x_data.detach().cpu().numpy() if isinstance(x_data, torch.Tensor) else np.asarray(x_data)
    y_np = np.asarray(y_labels).astype(int)

    classifier = _art_classifier(
        model=model,
        x_data=x_np,
        nb_classes=nb_classes,
        model_type=model_type,
        device=device,
        clip_values=clip_values
    )

    if attack_name == 'fgsm':
        attack = FastGradientMethod(estimator=classifier, **attack_params)
    elif attack_name == 'pgd':
        attack = ProjectedGradientDescent(estimator=classifier, **attack_params)
    elif attack_name == 'square':
        attack = SquareAttack(estimator=classifier, **attack_params)
    elif attack_name == 'hsj':
        attack = HopSkipJump(classifier=classifier, **attack_params)
    elif attack_name == 'simba':
        attack = SimBA(classifier=classifier, **attack_params)
    else:
        raise ValueError(f'Unsupported attack_name: {attack_name}')

    x_adv = attack.generate(x=x_np, y=y_np)
    return np.asarray(x_adv, dtype=np.float32)


def _rgr_curve_adversarial_core(
    model,
    x_data,
    prob_original,
    attack_strengths,
    model_class_order,
    class_order,
    y_true,
    attack_name: AttackName = 'fgsm',
    base_attack_params=None,
    class_weights=None,
    model_type='sklearn',
    device=None,
    rga_full=None,
    model_name='Model',
    plot=True,
    fig_size=(10, 6),
    verbose=True,
    save_path=None
):
    """
    Evaluate RGR robustness under adversarial perturbations.
    """
    prob_original = np.asarray(prob_original)
    attack_strengths = np.asarray(attack_strengths, dtype=float)
    model_class_order = np.asarray(model_class_order)
    class_order = np.asarray(class_order)
    y_true = np.asarray(y_true).astype(int)

    prob_original_aligned = align_proba_to_class_order(
        prob_original,
        model_class_order,
        class_order
    )

    n_classes = prob_original_aligned.shape[1]

    adv_rgr_scores = []
    per_class_rgr_list = []

    if verbose:
        print(f'Adversarial RGR Evaluation: {model_name}')
        print(f'Attack: {attack_name}')
        print(f'Testing {len(attack_strengths)} attack strengths')

    for eps in attack_strengths:
        params = {} if base_attack_params is None else dict(base_attack_params)

        if attack_name in ['fgsm', 'pgd', 'square']:
            params['eps'] = float(eps)

        elif attack_name == 'simba':
            params['epsilon'] = float(eps)

        elif attack_name == 'hsj':
            params['max_iter'] = int(eps)

        if attack_name == 'pgd' and 'eps_step' not in params:
            params['eps_step'] = max(float(eps) / 4.0, 1e-4)

        x_adv = _generate_adversarial_examples(
            model=model,
            x_data=x_data,
            y_labels=y_true,
            nb_classes=n_classes,
            attack_name=attack_name,
            attack_params=params,
            model_type=model_type,
            device=device
        )

        prob_perturbed_raw = _predict_probabilities(
            model,
            x_adv,
            model_type=model_type,
            device=device
        )

        prob_perturbed = align_proba_to_class_order(
            prob_perturbed_raw,
            model_class_order,
            class_order
        )

        rgr_val, rgr_per_class, _ = _rgr_cramer_multiclass(
            prob_original_aligned,
            prob_perturbed,
            class_order=class_order,
            class_weights=class_weights,
            verbose=False
        )

        rgr_value = nan_to_zero(rgr_val)
        adv_rgr_scores.append(rgr_value)
        per_class_rgr_list.append(rgr_per_class)

        if verbose:
            print(f'eps = {float(eps):.4f}: RGR = {rgr_value:.4f}')

    adv_rgr_scores = np.asarray(adv_rgr_scores, dtype=float)
    per_class_rgr_list = np.asarray(per_class_rgr_list, dtype=float)

    rgr_rescaled = rescale_by_rga(adv_rgr_scores, rga_full)
    aurgr = area_under_normalized_curve(attack_strengths, rgr_rescaled)

    result = {
        'method': 'adversarial',
        'attack_name': attack_name,
        'rgr_scores': adv_rgr_scores,
        'rgr_rescaled': rgr_rescaled,
        'aurgr': aurgr,
        'attack_strengths': attack_strengths,
        'strengths': attack_strengths,
        'per_class_rgr': per_class_rgr_list,
        'class_order': class_order
    }

    if verbose:
        print(f'Adversarial AURGR: {aurgr:.4f}')

    if plot or save_path is not None:
        plot_rgr(
            result,
            model_name=model_name,
            x_key='attack_strengths',
            x_label='Attack strength ε',
            title=f'Adversarial RGR Curve: {model_name}',
            fig_size=fig_size,
            save_path=save_path,
            show=(plot and save_path is None)
        )

    return result

def _compare_rgr_wasserstein_images_core(
    *,
    models,
    images,
    y_true,
    attack_model,
    preprocess_fn,
    attack_strengths,
    class_order,
    rga_dict=None,
    class_weights=None,
    verbose=True,
    max_iter=50,
    eps_step=0.01
):
    """Internal Wasserstein image-level comparison without plotting."""
    attack_strengths = np.asarray(attack_strengths, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    class_order = np.asarray(class_order)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attack_model = attack_model.to(device)
    attack_model.eval()

    images_tensor = images.detach().cpu() if isinstance(images, torch.Tensor) else torch.tensor(images, dtype=torch.float32)
    images_01 = torch.clamp((images_tensor + 1.0) / 2.0, 0.0, 1.0)

    class AttackWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            x_norm = x * 2.0 - 1.0
            return self.base_model(x_norm)

    wrapped_model = AttackWrapper(attack_model).to(device)
    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=nn.CrossEntropyLoss(),
        input_shape=tuple(images_01.shape[1:]),
        nb_classes=len(class_order),
        optimizer=None,
        clip_values=(0.0, 1.0),
        device_type='gpu' if str(device).startswith('cuda') and torch.cuda.is_available() else 'cpu'
    )

    results: dict[str, dict[str, Any]] = {
        name: {
            'attack_strengths': attack_strengths,
            'strengths': attack_strengths,
            'rgr_scores': [],
            'rgr_rescaled': [],
            'per_class_rgr': [],
            'class_order': class_order,
            'method': 'wasserstein_images',
            'aurgr': np.nan
        }
        for name in models
    }

    if verbose:
        print('Wasserstein image-level RGR')
        print(f'Testing {len(attack_strengths)} Wasserstein strengths')

    for eps in attack_strengths:
        if verbose:
            print(f'\nGenerating Wasserstein adversarial images: eps={eps:.4f}')

        attack = Wasserstein(
            estimator=classifier,
            eps=float(eps),
            eps_step=float(eps_step),
            max_iter=int(max_iter),
            targeted=False,
            verbose=False
        )

        x_adv_01 = attack.generate(x=images_01.numpy().astype(np.float32), y=y_true)
        x_adv_norm = torch.tensor(x_adv_01, dtype=torch.float32) * 2.0 - 1.0
        x_adv_features = preprocess_fn(x_adv_norm)

        for model_name, model_config in models.items():
            model, prob_original, model_class_order, model_type, model_device = model_config
            prob_original_aligned = align_proba_to_class_order(prob_original, model_class_order, class_order)

            prob_perturbed_raw = _predict_probabilities(
                model,
                x_adv_features,
                model_type=model_type,
                device=(model_device if model_device is not None else device)
            )
            prob_perturbed = align_proba_to_class_order(prob_perturbed_raw, model_class_order, class_order)

            rgr_val, rgr_per_class, _ = _rgr_cramer_multiclass(
                prob_original_aligned,
                prob_perturbed,
                class_order=class_order,
                class_weights=class_weights,
                verbose=False
            )

            rgr_value = nan_to_zero(rgr_val)
            results[model_name]['rgr_scores'].append(rgr_value)
            results[model_name]['per_class_rgr'].append(rgr_per_class)

            if verbose:
                print(f'{model_name}: RGR={rgr_value:.4f}')

    for model_name, result in results.items():
        rgr_scores = np.asarray(result['rgr_scores'], dtype=float)
        rga_full = rga_dict.get(model_name) if rga_dict else None
        rgr_rescaled = rescale_by_rga(rgr_scores, rga_full)
        aurgr = area_under_normalized_curve(attack_strengths, rgr_rescaled)

        result['rgr_scores'] = rgr_scores
        result['rgr_rescaled'] = rgr_rescaled
        result['per_class_rgr'] = np.asarray(result['per_class_rgr'], dtype=float)
        result['aurgr'] = float(aurgr)

    return results


def _compare_rgr_spatial_images_core(
    *,
    models,
    images,
    y_true,
    attack_model,
    preprocess_fn,
    attack_strengths,
    class_order,
    rga_dict=None,
    class_weights=None,
    verbose=True,
    num_translations=3,
    num_rotations=3
):
    """Internal spatial image-level comparison without plotting."""
    attack_strengths = np.asarray(attack_strengths, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    class_order = np.asarray(class_order)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attack_model = attack_model.to(device)
    attack_model.eval()

    images_tensor = images.detach().cpu() if isinstance(images, torch.Tensor) else torch.tensor(images, dtype=torch.float32)
    images_01 = torch.clamp((images_tensor + 1.0) / 2.0, 0.0, 1.0)

    class AttackWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            x_norm = x * 2.0 - 1.0
            return self.base_model(x_norm)

    wrapped_model = AttackWrapper(attack_model).to(device)
    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=nn.CrossEntropyLoss(),
        input_shape=tuple(images_01.shape[1:]),
        nb_classes=len(class_order),
        optimizer=None,
        clip_values=(0.0, 1.0),
        device_type='gpu' if str(device).startswith('cuda') and torch.cuda.is_available() else 'cpu'
    )

    results: dict[str, dict[str, Any]] = {
        name: {
            'attack_strengths': attack_strengths,
            'strengths': attack_strengths,
            'rgr_scores': [],
            'rgr_rescaled': [],
            'per_class_rgr': [],
            'class_order': class_order,
            'method': 'spatial_images',
            'aurgr': np.nan
        }
        for name in models
    }

    if verbose:
        print('Spatial image-level RGR')
        print(f'Testing {len(attack_strengths)} spatial strengths')

    for strength in attack_strengths:
        if verbose:
            print(f'\nGenerating spatial adversarial images: strength={strength:.4f}')

        attack = SpatialTransformation(
            classifier=classifier,
            max_translation=float(strength),
            max_rotation=float(strength),
            num_translations=int(num_translations),
            num_rotations=int(num_rotations),
            verbose=False
        )

        x_adv_01 = attack.generate(x=images_01.numpy().astype(np.float32), y=y_true)
        x_adv_norm = torch.tensor(x_adv_01, dtype=torch.float32) * 2.0 - 1.0
        x_adv_features = preprocess_fn(x_adv_norm)

        for model_name, model_config in models.items():
            model, prob_original, model_class_order, model_type, model_device = model_config
            prob_original_aligned = align_proba_to_class_order(prob_original, model_class_order, class_order)

            prob_perturbed_raw = _predict_probabilities(
                model,
                x_adv_features,
                model_type=model_type,
                device=(model_device if model_device is not None else device)
            )
            prob_perturbed = align_proba_to_class_order(prob_perturbed_raw, model_class_order, class_order)

            rgr_val, rgr_per_class, _ = _rgr_cramer_multiclass(
                prob_original_aligned,
                prob_perturbed,
                class_order=class_order,
                class_weights=class_weights,
                verbose=False
            )

            rgr_value = nan_to_zero(rgr_val)
            results[model_name]['rgr_scores'].append(rgr_value)
            results[model_name]['per_class_rgr'].append(rgr_per_class)

            if verbose:
                print(f'{model_name}: RGR={rgr_value:.4f}')

    for model_name, result in results.items():
        rgr_scores = np.asarray(result['rgr_scores'], dtype=float)
        rga_full = rga_dict.get(model_name) if rga_dict else None
        rgr_rescaled = rescale_by_rga(rgr_scores, rga_full)
        aurgr = area_under_normalized_curve(attack_strengths, rgr_rescaled)

        result['rgr_scores'] = rgr_scores
        result['rgr_rescaled'] = rgr_rescaled
        result['per_class_rgr'] = np.asarray(result['per_class_rgr'], dtype=float)
        result['aurgr'] = float(aurgr)

    return results


# ---- Private helpers ----
def _prepare_x_for_model(x_data, *, model_type):
    if model_type == 'pytorch':
        if isinstance(x_data, torch.Tensor):
            return x_data.float()
        return torch.tensor(x_data, dtype=torch.float32)

    return np.asarray(x_data, dtype=float)


def _predict_probabilities(model, x_data, *, model_type='sklearn', device=None):
    if model_type == 'sklearn':
        return model.predict_proba(x_data)

    if model_type == 'pytorch':
        if device is None:
            device = next(model.parameters()).device

        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data, dtype=torch.float32, device=device)
        else:
            x_data = x_data.to(device)

        model.eval()
        with torch.no_grad():
            logits = model(x_data)
            return torch.softmax(logits, dim=1).detach().cpu().numpy()

    raise ValueError(f"model_type must be 'sklearn' or 'pytorch', got '{model_type}'.")


def _add_gaussian_noise(x_data, sigma, *, model_type, rng):
    if model_type == 'pytorch':
        return x_data + torch.randn_like(x_data) * float(sigma)

    noise = rng.normal(0.0, float(sigma), size=x_data.shape)
    return x_data + noise


def _resolve_y_true_for_model(model_name, *, y_true=None, y_true_dict=None):
    if y_true_dict is not None:
        if model_name not in y_true_dict:
            raise ValueError(f'Missing true labels for model: {model_name}')
        return y_true_dict[model_name]

    if y_true is None:
        raise ValueError('y_true or y_true_dict is required for adversarial RGR.')

    return y_true


def _is_single_rgr_result(result):
    return isinstance(result, dict) and 'aurgr' in result and (
        'noise_levels' in result or 'attack_strengths' in result or 'strengths' in result
    )


def _infer_x_key(result):
    if 'noise_levels' in result:
        return 'noise_levels'
    if 'attack_strengths' in result:
        return 'attack_strengths'
    if 'strengths' in result:
        return 'strengths'
    raise ValueError('Cannot infer x-axis key from result.')


def _default_x_label(x_key):
    if x_key == 'noise_levels':
        return 'Noise Standard Deviation'
    if x_key == 'attack_strengths':
        return 'Attack Strength'
    return 'Strength'


def _curve_label(model_name, result):
    aurgr = result.get('aurgr', np.nan)
    method = result.get('method', None)

    if np.isfinite(aurgr):
        if method is not None:
            return f'{model_name} (AURGR={aurgr:.3f}, {method})'
        return f'{model_name} (AURGR={aurgr:.3f})'

    return f'{model_name}'


def _print_comparison_summary(results, *, metric_name='AURGR'):
    print('Robustness Comparison Summary')

    names = list(results.keys())
    scores = np.asarray([results[name].get('aurgr', np.nan) for name in names], dtype=float)

    for name, score in zip(names, scores):
        print(f'{name}: {metric_name} = {score:.4f}')

    finite = np.isfinite(scores)
    if len(names) >= 2 and np.any(finite):
        finite_idx = np.where(finite)[0]
        best_idx = finite_idx[int(np.argmax(scores[finite]))]
        worst_idx = finite_idx[int(np.argmin(scores[finite]))]

        print(f'Best: {names[best_idx]} ({metric_name}={scores[best_idx]:.4f})')
        print(f'Worst: {names[worst_idx]} ({metric_name}={scores[worst_idx]:.4f})')