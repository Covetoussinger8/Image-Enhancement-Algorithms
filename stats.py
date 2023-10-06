"""
    Statistical metrics library
"""

import numpy as np
from enhancement_algorithms import get_grayscale_image

def compute_mse(original_image_path, enhanced_image_path):
    """

    :param original_image_path:
    :param enhanced_image_path:
    :return:
    """
    original_image_array = np.asarray(get_grayscale_image(original_image_path))
    enhanced_image_array = np.asarray(get_grayscale_image(enhanced_image_path))

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
