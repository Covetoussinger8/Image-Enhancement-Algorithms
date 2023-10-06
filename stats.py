"""
    Statistical metrics library
"""

import numpy as np
from skimage import restoration
from enhancement_algorithms import get_grayscale_image_array


def compute_mse(original_image_path, enhanced_image_path):
    """

    :param original_image_path:
    :param enhanced_image_path:
    :return:
    """
    original_image_array = get_grayscale_image_array(original_image_path)
    enhanced_image_array = get_grayscale_image_array(enhanced_image_path)

    squared_differences = (original_image_array - enhanced_image_array) ** 2
    mean_squared_error = np.mean(squared_differences)

    return mean_squared_error


def compute_rmse(original_image_path, enhanced_image_path):
    """

    :param original_image_path:
    :param enhanced_image_path:
    :return:
    """
    mean_squared_error = compute_mse(original_image_path, enhanced_image_path)
    root_mean_squared_error = np.sqrt(mean_squared_error)

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
