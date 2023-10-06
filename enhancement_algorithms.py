"""
    Enhancement algorithms
"""

import time
import numpy as np
from PIL import Image
from skimage import exposure, restoration


def get_grayscale_image_array(image_path):
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
        return np.asarray(grayscale_image)
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None


def histogram_equalization(image_path):
    """

    :param image_path:
    :return:
    """
    start_time = time.monotonic_ns()

    image_array = get_grayscale_image_array(image_path)
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
    sigma_color = float(input("Choose a value for 'sigma_color': "))
    sigma_spatial = int(input("Choose a value for 'sigma_spatial': "))
    start_time = time.monotonic_ns()

    image_array = get_grayscale_image_array(image_path)
    filtered_image_array = restoration.denoise_bilateral(
                                                image=image_array,
                                                sigma_color=sigma_color,
                                                sigma_spatial=sigma_spatial)
    filtered_image = Image.fromarray((filtered_image_array * 255).astype(
        np.uint8))

    end_time = time.monotonic_ns()
    runtime_ns = end_time - start_time
    runtime_milliseconds = runtime_ns / 1e6

    return filtered_image, runtime_milliseconds


def wavelet_denoising(image_path):
    """

    :param image_path:
    :return:
    """
    wavelet = input("Choose a value for 'wavelet': ")
    start_time = time.monotonic_ns()

    image_array = get_grayscale_image_array(image_path)
    denoised_image_array = restoration.denoise_wavelet(image=image_array,
                                                       method='BayesShrink',
                                                       wavelet=wavelet)
    denoised_image = Image.fromarray((denoised_image_array * 255).astype(
        np.uint8))

    end_time = time.monotonic_ns()
    runtime_ns = end_time - start_time
    runtime_milliseconds = runtime_ns / 1e6

    return denoised_image, runtime_milliseconds


def total_variation_denoising(image_path):
    """

    :param image_path:
    :return:
    """
    weight = float(input("Choose a value for 'weight': "))
    start_time = time.monotonic_ns()

    image_array = get_grayscale_image_array(image_path)
    denoised_image_array = restoration.denoise_tv_chambolle(image=image_array,
                                                            weight=weight)
    denoised_image = Image.fromarray((denoised_image_array * 255).astype(
        np.uint8))

    end_time = time.monotonic_ns()
    runtime_ns = end_time - start_time
    runtime_milliseconds = runtime_ns / 1e6

    return denoised_image, runtime_milliseconds
