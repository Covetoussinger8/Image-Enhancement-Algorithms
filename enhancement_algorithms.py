"""
    Enhancement algorithms
"""

import time
import numpy as np
from PIL import Image
from skimage import exposure, restoration


def get_grayscale_image(image_path):
    """

    :param image_path:
    :return:
    """
    try:
        with Image.open(image_path) as image:
            if image.mode == 'L':  # Check whether the image is grayscale
                grayscale_image = image.copy()
            else:  # Else convert it to grayscale
                grayscale_image = image.convert('L')
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        grayscale_image = None

    return grayscale_image


def histogram_equalization(image_path):
    """

    :param image_path:
    :return:
    """
    start_time = time.monotonic_ns()

    image_array = np.asarray(get_grayscale_image(image_path))
    equalized_image_array = exposure.equalize_hist(image_array)
    equalized_image = Image.fromarray((equalized_image_array * 255).astype(
        np.uint8))

    end_time = time.monotonic_ns()
    runtime_ns = end_time - start_time
    runtime_milliseconds = runtime_ns / 1e6

    return equalized_image, runtime_milliseconds


def bilateral_filtering(image_path):
    """

    :param image_path:
    :return:
    """
    start_time = time.monotonic_ns()

    image_array = np.asarray(get_grayscale_image(image_path))
    denoised_image_array = restoration.denoise_bilateral(image=image_array,
                                                         sigma_color=0.028,
                                                         sigma_spatial=10,
                                                         channel_axis=None)
    equalized_image = Image.fromarray((denoised_image_array * 255).astype(
        np.uint8))

    end_time = time.monotonic_ns()
    runtime_ns = end_time - start_time
    runtime_milliseconds = runtime_ns / 1e6

    return equalized_image, runtime_milliseconds
