"""
Shared utilities for SAFE-AI metrics.

This module contains helpers used across RGA, RGR, and RGE,
plus optional image, Grad-CAM, dataset, and visualization utilities used by
image-based workflows.
"""

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from sklearn.metrics import auc
from torch.utils.data import Dataset


__all__ = [
    # Probability helpers
    'ensure_prob_matrix',
    'align_proba_to_class_order',
    'get_model_probabilities',
    'get_predictions_from_features',

    # Shared metric helpers
    'clean_pair',
    'validate_method',
    'validate_class_weights',
    'rescale_by_rga',
    'area_under_normalized_curve',
    'nan_to_zero',
    'resolve_class_orders',

    # Feature masking helpers
    'apply_feature_baseline',
    'mask_columns',
    'normalize_rankings',

    # RGA helpers
    'fill_nan_tail',
    'aurga_from_curve',
    'ideal_prob_matrix',

    # Image / Grad-CAM helpers
    'ScaledLinearHead',
    'CAMModel',
    'GradCAM',
    'train_cam_model',
    'blur_images_gaussian',
    'compute_gradcam_maps',
    'precompute_patch_rankings',
    'apply_importance_masking',
    'apply_patch_occlusion',
    'extract_features_from_images',

    # Dataset / visualization helpers
    'crop_img',
    'CroppedImage',
    'denorm_img',
    'show_heatmap_per_class',
    'show_occlusions_same_idx'
]


# ---- Probability helpers ----
def ensure_prob_matrix(prob, class_order):
    """
    Ensure that probabilities are represented as a 2D probability matrix.

    Parameters
    ----------
    prob : array-like
        Either a 1D binary positive-class probability vector or a 2D
        probability matrix.

    class_order : array-like
        Class order corresponding to the probability columns.

    Returns
    -------
    np.ndarray
        Probability matrix with shape (n_samples, n_classes).
    """
    prob = np.asarray(prob, dtype=float)
    class_order = np.asarray(class_order)

    if prob.ndim == 1:
        if len(class_order) != 2:
            raise ValueError('1D prob is only supported for binary (2 classes).')
        p_pos = prob
        return np.column_stack([1.0 - p_pos, p_pos])

    if prob.ndim == 2:
        if prob.shape[1] != len(class_order):
            raise ValueError(
                f'prob has {prob.shape[1]} columns but class_order has {len(class_order)}.'
            )
        return prob

    raise ValueError('prob must be shape (n,) or (n,c).')


def align_proba_to_class_order(prob, model_class_order, target_class_order):
    """
    Align probability matrix columns to match a target class order.

    Parameters
    ----------
    prob : array-like, shape (n_samples, n_classes)
        Probability matrix with columns in model_class_order.

    model_class_order : array-like
        Current order of probability columns, for example ``model.classes_``.

    target_class_order : array-like
        Desired output class order.

    Returns
    -------
    np.ndarray
        Probability matrix with columns reordered to target_class_order.
    """
    prob = np.asarray(prob)
    model_class_order = list(model_class_order)
    target_class_order = list(target_class_order)

    idx = [model_class_order.index(c) for c in target_class_order]
    return prob[:, idx]


def get_model_probabilities(model: Any, x, class_order=None):
    """
    Get predicted probabilities from a sklearn estimator or Pipeline.

    Parameters
    ----------
    model : object
        Fitted sklearn estimator or Pipeline with predict_proba.

    x : array-like or DataFrame
        Input data.

    class_order : array-like, optional
        Desired class order. If provided, probabilities are aligned using
        ``model.classes_``.

    Returns
    -------
    np.ndarray
        Probability matrix.
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError('Model must support predict_proba().')

    prob = model.predict_proba(x)

    if class_order is not None:
        if not hasattr(model, 'classes_'):
            raise ValueError('class_order was provided, but model has no classes_ attribute.')

        prob = align_proba_to_class_order(
            prob,
            model_class_order=model.classes_,
            target_class_order=class_order
        )

    return prob


def get_predictions_from_features(
    features,
    model: Any,
    model_class_order,
    class_order,
    model_type='sklearn',
    device=None,
    batch_size=64
):
    """
    Get class probabilities from a model given feature vectors.

    Parameters
    ----------
    features : array-like
        Feature matrix with shape (n_samples, n_features).

    model : object
        sklearn model with predict_proba or PyTorch module producing logits.

    model_class_order : array-like
        Class order produced by the model.

    class_order : array-like
        Desired canonical class order.

    model_type : {'sklearn', 'pytorch'}, default='sklearn'
        Type of model.

    device : torch.device or str, optional
        Device used for PyTorch inference.

    batch_size : int, default=64
        Batch size for PyTorch inference.

    Returns
    -------
    np.ndarray
        Probabilities aligned to class_order.
    """
    if model_type == 'sklearn':
        probs = model.predict_proba(features)

    elif model_type == 'pytorch':
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = torch.tensor(features[i:i + batch_size], dtype=torch.float32, device=device)
                logits = model(batch)
                probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.vstack(probs_list)

    else:
        raise ValueError(f"model_type must be 'sklearn' or 'pytorch', got {model_type}")

    return align_proba_to_class_order(probs, model_class_order, class_order)


# ---- Shared metric helpers ----
def clean_pair(a, b):
    """
    Clean two paired 1D arrays by removing non-finite paired values.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    if len(a) != len(b):
        raise ValueError(f'Inputs must have the same length. Got {len(a)} and {len(b)}.')

    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def validate_method(method, *, allowed):
    """
    Validate that a method name is one of the allowed options.
    """
    if method not in allowed:
        raise ValueError(f'method must be one of {allowed}. Got {method}.')


def validate_class_weights(class_weights, n_classes):
    """
    Validate or create class weights for multiclass aggregation.
    """
    if class_weights is None:
        return np.ones(n_classes, dtype=float) / n_classes

    class_weights = np.asarray(class_weights, dtype=float)
    if len(class_weights) != n_classes:
        raise ValueError(
            f'class_weights length {len(class_weights)} does not match n_classes {n_classes}.'
        )
    return class_weights


def rescale_by_rga(scores, rga_full):
    """
    Rescale a metric curve by a full RGA score if provided.
    """
    scores = np.asarray(scores, dtype=float)
    if rga_full is not None and np.isfinite(rga_full):
        return scores * float(rga_full)
    return scores


def area_under_normalized_curve(x_values, y_values):
    """
    Compute area under a curve after normalizing x-values to [0, 1].
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if len(x_values) == 0:
        return np.nan

    max_x = float(np.max(x_values))
    x_norm = x_values / max_x if max_x > 0 else x_values
    return float(auc(x_norm, y_values))


def nan_to_zero(value):
    """
    Replace a non-finite scalar value with 0.0.
    """
    return 0.0 if not np.isfinite(value) else float(value)


def resolve_class_orders(model, *, model_class_order=None, class_order=None, prob=None):
    """
    Resolve the model output class order and target class order.
    """
    if model_class_order is None:
        if hasattr(model, 'classes_'):
            model_class_order = np.asarray(model.classes_)
        elif class_order is not None:
            model_class_order = np.asarray(class_order)
        elif prob is not None and np.asarray(prob).ndim == 2:
            model_class_order = np.arange(np.asarray(prob).shape[1])
        else:
            raise ValueError('model_class_order or class_order must be provided.')

    model_class_order = np.asarray(model_class_order)

    if class_order is None:
        class_order = model_class_order
    else:
        class_order = np.asarray(class_order)

    return model_class_order, class_order


# ---- Feature masking helpers ----
def apply_feature_baseline(x_masked, cols, *, baseline, feat_mean=None):
    """
    Apply a feature masking baseline in-place.
    """
    if baseline == 'zero':
        x_masked[:, cols] = 0.0
    elif baseline == 'mean':
        if feat_mean is None:
            raise ValueError("feat_mean is required when baseline='mean'.")
        x_masked[:, cols] = feat_mean[cols]
    else:
        raise ValueError(f"Unknown baseline: {baseline}. Use 'zero' or 'mean'.")


def mask_columns(x, cols, *, baseline, feat_mean=None):
    """
    Return a copy of x with selected columns masked.
    """
    x_masked = np.asarray(x, dtype=float).copy()

    if len(cols) > 0:
        apply_feature_baseline(
            x_masked,
            cols,
            baseline=baseline,
            feat_mean=feat_mean
        )

    return x_masked


def normalize_rankings(feature_rankings, models):
    """
    Normalize feature ranking input into a model_name -> ranking mapping.
    """
    if isinstance(feature_rankings, dict):
        return feature_rankings
    if feature_rankings is None:
        return {}
    return {name: feature_rankings for name in models}


# ---- Image / Grad-CAM helpers ----
class ScaledLinearHead(nn.Module):
    """
    Linear head that optionally applies the same scaler as sklearn models.
    """

    def __init__(self, in_dim, n_classes, scaler=None, eps=1e-12):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
        self.has_scaler = scaler is not None

        if self.has_scaler:
            mean = torch.tensor(scaler.mean_, dtype=torch.float32)
            scale = torch.tensor(scaler.scale_, dtype=torch.float32)
            scale = torch.clamp(scale, min=eps)
            self.register_buffer('mean', mean)
            self.register_buffer('scale', scale)

    def forward(self, feats):
        if self.has_scaler:
            feats = (feats - self.mean) / self.scale
        return self.linear(feats)


class CAMModel(nn.Module):
    """
    Simple wrapper combining a feature extractor and classification head.
    """

    def __init__(self, feature_extractor, head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, x):
        feats = self.feature_extractor(x)
        return self.head(feats)


class GradCAM:
    """
    Grad-CAM for a CAMModel.
    """

    def __init__(self, cam_model, target_layer=None):
        self.model = cam_model

        if target_layer is None:
            fe = cam_model.feature_extractor
            if hasattr(fe, 'layer4'):
                target_layer = fe.layer4[-1].conv2
            else:
                raise ValueError('Cannot auto-detect target layer. Provide target_layer.')

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def close(self):
        if getattr(self, '_fwd_handle', None) is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None

        if getattr(self, '_bwd_handle', None) is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    def _save_activation(self, _module, _inp, out):
        self.activations = out

    def _save_gradient(self, _module, _grad_inp, grad_out):
        self.gradients = grad_out[0]

    @torch.no_grad()
    def predict_classes(self, images, device, batch_size=64):
        self.model.eval()
        preds = []
        for i in range(0, len(images), batch_size):
            x = images[i:i + batch_size].to(device, non_blocking=True)
            logits = self.model(x)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)

    def cam_single(self, image, target_class=None, device=None):
        if device is None:
            device = next(self.model.parameters()).device

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(device, non_blocking=True)
        image.requires_grad_(True)

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations, self.gradients = None, None

        logits = self.model(image)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError('GradCAM hooks did not capture activations or gradients.')

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = nnf.relu(cam)

        cam = nnf.interpolate(cam, size=image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        mn, mx = float(cam.min()), float(cam.max())
        if mx > mn:
            return (cam - mn) / (mx - mn)
        return np.zeros_like(cam)


def train_cam_model(
    feature_extractor,
    images,
    labels,
    scaler=None,
    n_classes=None,
    device=None,
    epochs=15,
    lr=1e-3,
    batch_size=64,
    verbose=True
):
    """
    Train a linear CAM head on top of a frozen feature extractor.
    """
    if device is None:
        device = next(feature_extractor.parameters()).device

    feature_extractor.eval().to(device)
    for p in feature_extractor.parameters():
        p.requires_grad_(False)

    labels = np.asarray(labels)
    if n_classes is None:
        n_classes = int(len(np.unique(labels)))

    if verbose:
        print('Extracting raw features for CAM training...')

    feats_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = images[i:i + batch_size].to(device, non_blocking=True)
            feats_list.append(feature_extractor(x).cpu().numpy())
    feats = np.vstack(feats_list)

    head = ScaledLinearHead(feats.shape[1], n_classes, scaler=scaler).to(device)
    cam_model = CAMModel(feature_extractor, head).to(device)

    x = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True
    )

    opt = torch.optim.Adam(cam_model.head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    if verbose:
        print(f'Training CAM head for {epochs} epochs...')

    for ep in range(epochs):
        cam_model.head.train()
        tot_loss, correct, total = 0.0, 0, 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = cam_model.head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            tot_loss += float(loss.item())
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))

        if verbose and ((ep + 1) % 5 == 0 or ep == epochs - 1):
            print(
                f'Epoch {ep + 1:02d}/{epochs}: '
                f'loss={tot_loss / len(loader):.4f}, acc={100 * correct / total:.2f}%'
            )

    cam_model.eval()
    return cam_model


def blur_images_gaussian(images, ksize=31, sigma=7.0):
    """
    Apply Gaussian blur to a batch of images using separable convolution.
    """
    if ksize % 2 == 0:
        ksize += 1

    device = images.device
    dtype = images.dtype

    x = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    g_x = g.view(1, 1, 1, ksize).repeat(images.shape[1], 1, 1, 1)
    g_y = g.view(1, 1, ksize, 1).repeat(images.shape[1], 1, 1, 1)

    pad = ksize // 2
    out = nnf.conv2d(images, g_x, padding=(0, pad), groups=images.shape[1])
    out = nnf.conv2d(out, g_y, padding=(pad, 0), groups=images.shape[1])
    return out


def compute_gradcam_maps(images, cam_model, device=None, batch_pred=64, verbose=True):
    """
    Compute Grad-CAM importance maps for a batch of images.
    """
    if device is None:
        device = next(cam_model.parameters()).device

    gradcam = GradCAM(cam_model)

    if verbose:
        print('Predicting target classes for Grad-CAM...')
    targets = gradcam.predict_classes(images, device=device, batch_size=batch_pred)

    if verbose:
        print('Computing Grad-CAM maps...')

    maps = []
    for i in range(len(images)):
        maps.append(gradcam.cam_single(images[i:i + 1], target_class=int(targets[i]), device=device))
        if verbose and (i + 1) % 100 == 0:
            print(f'{i + 1}/{len(images)} maps')

    gradcam.close()
    return np.asarray(maps, dtype=np.float32)


def precompute_patch_rankings(importance_maps, patch_size=32):
    """
    Convert per-pixel importance maps into per-image patch rankings.
    """
    n, h, w = importance_maps.shape
    n_ph = h // patch_size
    n_pw = w // patch_size

    patch_coords = []
    for ph in range(n_ph):
        for pw in range(n_pw):
            y0 = ph * patch_size
            x0 = pw * patch_size
            y1 = min(y0 + patch_size, h)
            x1 = min(x0 + patch_size, w)
            patch_coords.append((y0, y1, x0, x1))

    rankings = []
    for i in range(n):
        imp = importance_maps[i]
        scores = np.array(
            [imp[y0:y1, x0:x1].mean() for (y0, y1, x0, x1) in patch_coords],
            dtype=np.float32,
        )
        rankings.append(np.argsort(scores)[::-1])

    meta = {
        'patch_size': patch_size,
        'patch_coords': patch_coords,
        'total_patches': len(patch_coords),
        'n_patches_h': n_ph,
        'n_patches_w': n_pw,
    }
    return rankings, meta


def apply_importance_masking(
    images,
    patch_rankings,
    patch_meta,
    fraction_to_mask,
    mask_strategy='most_important',
    mask_value=0.0,
    baseline='constant',
    blur_ksize=31,
    blur_sigma=7.0
):
    """
    Mask a fraction of the image area using patch importance rankings.
    """
    out = images.clone()
    _, _, h, w = out.shape

    blurred = None
    if baseline == 'blur':
        blurred = blur_images_gaussian(images, ksize=blur_ksize, sigma=blur_sigma)

    patch_size = patch_meta['patch_size']
    patch_pixels = patch_size * patch_size
    total_pixels = h * w

    pixels_to_mask = int(fraction_to_mask * total_pixels)
    k = pixels_to_mask // patch_pixels
    k = min(k, patch_meta['total_patches'])
    if k <= 0:
        return out

    coords = patch_meta['patch_coords']

    for i in range(out.shape[0]):
        order = patch_rankings[i]
        if mask_strategy == 'most_important':
            chosen = order[:k]
        else:
            raise ValueError(f'Unknown mask_strategy: {mask_strategy}')

        for idx in chosen:
            y0, y1, x0, x1 = coords[int(idx)]
            if baseline == 'blur':
                out[i, :, y0:y1, x0:x1] = blurred[i, :, y0:y1, x0:x1]
            else:
                out[i, :, y0:y1, x0:x1] = mask_value

    return out


def apply_patch_occlusion(
    images,
    num_patches,
    patch_size=32,
    random_seed=None,
    mask_value=0.0,
    baseline='constant',
    blur_ksize=31,
    blur_sigma=7.0
):
    """
    Random patch masking for a batch of images.
    """
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    out = images.clone()
    _, _, h, w = out.shape
    if num_patches <= 0:
        return out

    blurred = None
    if baseline == 'blur':
        blurred = blur_images_gaussian(images, ksize=blur_ksize, sigma=blur_sigma)

    if h < patch_size or w < patch_size:
        raise ValueError('patch_size must not exceed image height or width.')

    for i in range(out.shape[0]):
        for _ in range(num_patches):
            y0 = random.randint(0, h - patch_size)
            x0 = random.randint(0, w - patch_size)
            if baseline == 'blur':
                out[i, :, y0:y0 + patch_size, x0:x0 + patch_size] = blurred[
                    i, :, y0:y0 + patch_size, x0:x0 + patch_size
                ]
            else:
                out[i, :, y0:y0 + patch_size, x0:x0 + patch_size] = mask_value

    return out


def extract_features_from_images(images, feature_extractor, pca=None, scaler=None, device=None, batch_size=64):
    """
    Extract features from images using a torch feature extractor.
    """
    feature_extractor.eval()
    if device is None:
        device = next(feature_extractor.parameters()).device

    feats_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device, non_blocking=True)
            feats_list.append(feature_extractor(batch).cpu().numpy())
    x = np.vstack(feats_list)

    if pca is not None:
        x = pca.transform(x)
    if scaler is not None:
        x = scaler.transform(x)

    return x


# ---- Dataset / visualization helpers ----
def crop_img(img):
    """
    Crop image to the bounding box of the largest foreground object.
    """
    import cv2

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if len(cnts) == 0:
        return img

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    add_pixels = 5
    height, width = img.shape[:2]

    x = max(0, x - add_pixels)
    y = max(0, y - add_pixels)
    w = min(width - x, w + 2 * add_pixels)
    h = min(height - y, h + 2 * add_pixels)

    new_img = img[y:y + h, x:x + w].copy()

    if new_img.shape[0] < 100 or new_img.shape[1] < 100:
        return img

    return new_img


class CroppedImage(Dataset):
    """
    PyTorch Dataset for loading images with optional automatic cropping.
    """

    def __init__(self, root_dir, transform=None, apply_crop=True):
        from torchvision import datasets

        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.apply_crop = apply_crop
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        import cv2
        from PIL import Image

        img_path, label = self.dataset.samples[idx]
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f'Could not read image: {img_path}')

        if self.apply_crop:
            try:
                img = crop_img(img)
            except Exception as exc:
                print(f'Cropping failed for {img_path}: {exc}')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


def denorm_img(img_t, mean=0.5, std=0.5):
    """
    Denormalize a normalized image tensor for visualization.
    """
    img = img_t.detach().cpu().float()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def show_heatmap_per_class(
    x_images,
    importance_maps,
    labels,
    class_names,
    n_classes,
    alpha=0.45,
    cmap='jet',
    save_path=None
):
    """
    Display Grad-CAM heatmap overlays for one sample from each class.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_classes, 2, figsize=(10, 5 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx, class_name in enumerate(class_names):
        idx = np.where(labels == class_idx)[0][0]

        img = denorm_img(x_images[idx])
        hm = np.clip(importance_maps[idx], 0, 1)

        axes[class_idx, 0].imshow(img)
        axes[class_idx, 0].set_title(f'{class_name} - Original')
        axes[class_idx, 0].axis('off')

        axes[class_idx, 1].imshow(img)
        axes[class_idx, 1].imshow(hm, alpha=alpha, cmap=cmap)
        axes[class_idx, 1].set_title(f'{class_name} - Grad-CAM')
        axes[class_idx, 1].axis('off')

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


def show_occlusions_same_idx(
    x_images,
    patch_rankings,
    patch_meta,
    idx=0,
    fractions=(0.0, 0.2, 0.4, 0.6, 0.8, 1),
    baseline='blur',
    blur_ksize=31,
    blur_sigma=7.0,
    n_cols=3,
    save_path=None
):
    """
    Visualize progressive occlusion of image regions based on patch rankings.
    """
    import matplotlib.pyplot as plt

    img0 = x_images[idx:idx + 1]
    n_rows = int(np.ceil(len(fractions) / n_cols))
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for j, frac in enumerate(fractions, 1):
        img_occ = apply_importance_masking(
            images=img0,
            patch_rankings=[patch_rankings[idx]],
            patch_meta=patch_meta,
            fraction_to_mask=frac,
            mask_strategy='most_important',
            baseline=baseline,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma
        )[0]

        ax = plt.subplot(n_rows, n_cols, j)
        ax.imshow(denorm_img(img_occ))
        ax.set_title(f'{int(frac * 100)}% occluded')
        ax.axis('off')

    plt.suptitle('Grad-CAM–guided occlusion', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


# ---- RGA curve helpers ----
def fill_nan_tail(vec):
    """
    Replace the first non-finite value and all following values with 0.0.
    """
    vec = np.asarray(vec, dtype=float).copy()
    bad = np.where(~np.isfinite(vec))[0]
    if len(bad) > 0:
        vec[bad[0]:] = 0.0
    return vec


def aurga_from_curve(curve):
    """
    Compute area under an RGA curve on a normalized [0, 1] x-axis.
    """
    curve = fill_nan_tail(curve)
    x = np.linspace(0, 1, len(curve))
    return float(auc(x, curve))


def ideal_prob_matrix(y_labels, class_order):
    """
    Build an ideal one-hot probability matrix from labels and class order.
    """
    y_labels = np.asarray(y_labels)
    class_order = np.asarray(class_order)
    ideal = np.zeros((len(y_labels), len(class_order)), dtype=np.float32)

    for k, c in enumerate(class_order):
        ideal[:, k] = np.equal(y_labels, c).astype(np.float32)

    return ideal