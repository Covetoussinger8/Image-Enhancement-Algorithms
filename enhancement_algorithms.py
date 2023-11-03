"""
    Enhancement algorithms
"""


import time
import numpy as np
from skimage import exposure, restoration



def histogram_equalization(image_array: np.ndarray):
    """
    :param image_array:
    :return:
    """
    start_time = time.monotonic_ns()

    equalized_image_array = exposure.equalize_hist(image_array)
    equalized_image_array = (equalized_image_array * 255).astype(np.uint8)

    end_time = time.monotonic_ns()
    runtime_ns = end_time - start_time
    runtime_milliseconds = runtime_ns / 1e6

    return equalized_image_array, runtime_milliseconds


def bilateral_filtering(image_array: np.ndarray):
    """
    :param image_array:
    :return:
    """
    start_time = time.monotonic_ns()

    filtered_image_array = restoration.denoise_bilateral(
                                                image=image_array,
                                                sigma_color=0.025,
                                                sigma_spatial=10)
    filtered_image_array = (filtered_image_array * 255).astype(np.uint8)

    end_time = time.monotonic_ns()
    runtime_ns = end_time - start_time
    runtime_milliseconds = runtime_ns / 1e6

    return filtered_image_array, runtime_milliseconds


def wavelet_denoising(image_array):
    """
    :param image_array:
    :return:
    """
    start_time = time.monotonic_ns()

    denoised_image_array = restoration.denoise_wavelet(image=image_array,
                                                       method='BayesShrink',
                                                       wavelet='db1')
    denoised_image_array = (denoised_image_array * 255).astype(np.uint8)

    end_time = time.monotonic_ns()
    runtime_ns = end_time - start_time
    runtime_milliseconds = runtime_ns / 1e6

    return denoised_image_array, runtime_milliseconds


def total_variation_denoising(image_array):
    """

    :param image_path:
    :return:
    """
    start_time = time.monotonic_ns()

    denoised_image_array = restoration.denoise_tv_chambolle(image=image_array)
    denoised_image_array = (denoised_image_array * 255).astype(np.uint8)

    end_time = time.monotonic_ns()
    runtime_ns = end_time - start_time
    runtime_milliseconds = runtime_ns / 1e6

    return denoised_image_array, runtime_milliseconds


def clahe(image_array):
    start_time = time.monotonic_ns()

    equalized_image_array = exposure.equalize_adapthist(image=image_array,
                                                        clip_limit=0.30,
                                                        kernel_size=(58, 58))
    equalized_image_array = (equalized_image_array * 255).astype(np.uint8)

    end_time = time.monotonic_ns()
    runtime_ns = end_time - start_time
    runtime_milliseconds = runtime_ns / 1e6

    return equalized_image_array, runtime_milliseconds
