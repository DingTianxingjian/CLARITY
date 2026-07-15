import torch
import torch.nn as nn
from skimage.measure import find_contours
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt, binary_erosion
import numpy as np


def compute_dice(pred, target):
    """Calculate the mean Dice Coefficient for multi-class data."""
    assert pred.shape == target.shape, "Prediction and target must have the same shape"

    smooth = 1e-6
    intersection = (pred & target).float().sum()  # Sum over height and width dimensions
    union = pred.float().sum() + target.float().sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.item()

def compute_nsd(pred, target, tolerance=1.0, spacing=None):
    """
    Compute Normalized Surface Dice (NSD) between binary prediction and target masks.

    Args:
        pred (torch.Tensor): Binary prediction mask (H, W) or (D, H, W).
        target (torch.Tensor): Binary ground truth mask (same shape as pred).
        tolerance (float): Distance tolerance in physical units.
        spacing (tuple or list, optional): Pixel spacing in each dimension. 
                                           Should match the number of dimensions.

    Returns:
        float: NSD score between 0 and 1.
    """
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    pred = pred.cpu().numpy().astype(np.bool_)
    target = target.cpu().numpy().astype(np.bool_)

    ndim = pred.ndim
    if spacing is None:
        spacing = [1.0] * ndim

    def get_surface(mask):
        # Surface = mask - eroded(mask)
        return mask ^ binary_erosion(mask)

    pred_surf = get_surface(pred)
    target_surf = get_surface(target)

    # Compute distance transforms
    dt_pred = distance_transform_edt(~pred_surf, sampling=spacing)
    dt_target = distance_transform_edt(~target_surf, sampling=spacing)

    # Distance of each surface point to the other surface
    pred_to_target_dist = dt_target[pred_surf]
    target_to_pred_dist = dt_pred[target_surf]

    # Count points within tolerance
    pred_within_tol = (pred_to_target_dist <= tolerance).sum()
    target_within_tol = (target_to_pred_dist <= tolerance).sum()

    # Total surface points
    total_surface_points = pred_surf.sum() + target_surf.sum()
    if total_surface_points == 0:
        return 1.0 if pred.sum() == target.sum() else 0.0

    nsd = (pred_within_tol + target_within_tol) / total_surface_points
    return float(nsd)


def compute_nsd_prev(pred, target, tolerance=1.0):
    """
    Compute Normalized Surface Dice (NSD) between two binary masks.
    
    Parameters:
    pred (numpy.ndarray): Binary predicted mask
    target (numpy.ndarray): Binary ground truth mask
    tolerance (float): The surface tolerance in pixels

    Returns:
    float: NSD score
    """
    # Find contours of predicted and ground truth masks
    pred_contours = find_contours(pred, level=0.5)
    target_contours = find_contours(target, level=0.5)

    # Convert contours to a list of points
    pred_points = np.vstack(pred_contours) if pred_contours else np.array([])
    target_points = np.vstack(target_contours) if target_contours else np.array([])

    if len(pred_points) == 0 or len(target_points) == 0:
        return 0.0  # If no valid contours, NSD is zero

    # Compute distance matrices
    dist_pred_to_target = cdist(pred_points, target_points)
    dist_target_to_pred = cdist(target_points, pred_points)

    # Check if points are within the tolerance
    pred_within_tol = np.any(dist_pred_to_target <= tolerance, axis=1).sum()
    target_within_tol = np.any(dist_target_to_pred <= tolerance, axis=1).sum()

    # Compute NSD
    nsd_score = (pred_within_tol + gt_within_tol) / (len(pred_points) + len(gt_points))

    return nsd_score
