"""
    Statistical metrics library
"""

import numpy as np
from skimage import restoration
from skimage.metrics import (mean_squared_error, peak_signal_noise_ratio,
                             structural_similarity)
from enhancement_algorithms import get_grayscale_image_array


def compute_mse(original_image_path, enhanced_image_path):
    """

    :param original_image_path:
    :param enhanced_image_path:
    :return:
    """
    original_image_array = get_grayscale_image_array(original_image_path)
    enhanced_image_array = get_grayscale_image_array(enhanced_image_path)

    mse = mean_squared_error(original_image_array, enhanced_image_array)

    return mse


def compute_rmse(original_image_path, enhanced_image_path):
    """

    :param original_image_path:
    :param enhanced_image_path:
    :return:
    """
    mse = compute_mse(original_image_path, enhanced_image_path)
    root_mean_squared_error = np.sqrt(mse)

    return root_mean_squared_error


def compute_cnr(enhanced_image_path, ground_truth_path):
    """

    :param enhanced_image_path:
    :param ground_truth_path:
    :return:
    """
    enhanced_image_array = get_grayscale_image_array(enhanced_image_path)
    ground_truth_array = get_grayscale_image_array(ground_truth_path)

    if enhanced_image_array is None or ground_truth_array is None:
        return None

    enhanced_image_array = enhanced_image_array.flatten()
    ground_truth_array = ground_truth_array.flatten()

    if np.all(ground_truth_array == 0):
        return 0.0

    # roi = region of interest
    roi_pixel_values = np.empty(0)
    background_pixel_values = np.empty(0)

    for enhanced_pixel, ground_truth_pixel in zip(enhanced_image_array,
                                                  ground_truth_array):
        if ground_truth_pixel == 255:
            roi_pixel_values.append(enhanced_pixel)
        else:
            background_pixel_values.append(enhanced_pixel)

    mean_roi = np.mean(roi_pixel_values)
    mean_background = np.mean(background_pixel_values)
    std_noise = restoration.estimate_sigma(background_pixel_values)

    cnr = (mean_roi - mean_background) / std_noise

    return cnr


def compute_ambe(original_image_path, enhanced_image_path):
    """

    :param original_image_path:
    :param enhanced_image_path:
    :return:
    """
    original_image_array = get_grayscale_image_array(original_image_path)
    enhanced_image_array = get_grayscale_image_array(enhanced_image_path)

    mean_brightness_original = np.mean(original_image_array)
    mean_brightness_enhanced = np.mean(enhanced_image_array)

    ambe = np.abs(mean_brightness_original - mean_brightness_enhanced)

    return ambe


def compute_psnr(original_image_path, enhanced_image_path):
    """

    :param original_image_path:
    :param enhanced_image_path:
    :return:
    """
    original_image_array = get_grayscale_image_array(original_image_path)
    enhanced_image_array = get_grayscale_image_array(enhanced_image_path)

    psnr = peak_signal_noise_ratio(original_image_array, enhanced_image_array)

    return psnr


def compute_ssim(original_image_path, enhanced_image_path):
    """

    :param original_image_path:
    :param enhanced_image_path:
    :return:
    """
    original_image_array = get_grayscale_image_array(original_image_path)
    enhanced_image_array = get_grayscale_image_array(enhanced_image_path)

    ssim = structural_similarity(original_image_array, enhanced_image_array)

    return ssim


def generate_csv(file_path, dataset):
    assert type(dataset) == list or type(dataset) == tuple

    df = pd.DataFrame(dataset)
    df.to_csv(file_path, index=False)