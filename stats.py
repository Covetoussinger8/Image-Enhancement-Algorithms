"""
    Statistical metrics library
"""

import numpy as np
import pandas as pd
from skimage import restoration
from skimage.metrics import (mean_squared_error, peak_signal_noise_ratio,
                             structural_similarity)


def get_roi_and_background(enhanced_image_array: np.ndarray,
                           ground_truth_array: np.ndarray):
    if enhanced_image_array is None or ground_truth_array is None:
        raise ValueError("Both input arrays must not be None")

    flat_enhanced_image_array = enhanced_image_array.flatten()
    flat_ground_truth_array = ground_truth_array.flatten()

    if np.all(ground_truth_array == 0):
        return 0.0

    roi_mask = (flat_ground_truth_array >= 122)

    # Use the mask to extract pixel values for the ROI and background
    roi_pixel_values = flat_enhanced_image_array[roi_mask]
    background_pixel_values = flat_enhanced_image_array[~roi_mask]

    return roi_pixel_values, background_pixel_values


def compute_mse(original_image_array: np.ndarray,
                enhanced_image_array: np.ndarray):
    """

    :param original_image_array:
    :param enhanced_image_array:
    :return:
    """
    mse = mean_squared_error(original_image_array, enhanced_image_array)

    return mse


def compute_rmse(original_image_array: np.ndarray,
                 enhanced_image_array: np.ndarray):
    """
    :param original_image_arrray:
    :param enhanced_image_array:
    :return:
    """
    mse = compute_mse(original_image_array, enhanced_image_array)
    root_mean_squared_error = np.sqrt(mse)

    return root_mean_squared_error


def compute_cnr(enhanced_image_array: np.ndarray,
                ground_truth_array: np.ndarray):
    roi_values, background_values = get_roi_and_background(enhanced_image_array,
                                                           ground_truth_array)

    mean_roi = np.mean(roi_values)
    mean_background = np.mean(background_values)
    std_background = np.std(background_values)

    cnr = np.abs(mean_roi - mean_background) / std_background

    return cnr


def compute_gcnr(enhanced_image_array: np.ndarray, ground_truth_array:
np.ndarray):
    roi_values, background_values = get_roi_and_background(enhanced_image_array,
                                                           ground_truth_array)

    roi_hist, _ = np.histogram(roi_values, bins=256, density=True)
    background_hist, _ = np.histogram(background_values, bins=256, density=True)

    ovl = 0.0

    for roi_prob, background_prob in zip(roi_hist, background_hist):
        ovl += min(roi_prob, background_prob)

    gcnr = 1 - ovl

    return gcnr

def compute_ambe(original_image_array: np.ndarray,
                 enhanced_image_array: np.ndarray):
    """

    :param original_image_array:
    :param enhanced_image_array:
    :return:
    """
    mean_brightness_original = np.mean(original_image_array)
    mean_brightness_enhanced = np.mean(enhanced_image_array)

    ambe = np.abs(mean_brightness_original - mean_brightness_enhanced)

    return ambe


def compute_psnr(original_image_array, enhanced_image_array):
    """

    :param original_image_array:
    :param enhanced_image_array:
    :return:
    """
    psnr = peak_signal_noise_ratio(original_image_array, enhanced_image_array)

    return psnr


def compute_ssim(ground_truth_array, enhanced_image_array):
    """

    :param original_image_array:
    :param enhanced_image_array:
    :return:
    """
    ssim = structural_similarity(ground_truth_array, enhanced_image_array)

    return ssim


def generate_csv(csv_directory, dataset):
    assert type(dataset) == list or type(dataset) == tuple

    df = pd.DataFrame(dataset)
    df.to_csv(str(csv_directory / "stats.csv"), index=False)