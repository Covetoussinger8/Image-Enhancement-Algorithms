"""
    describe the program
"""

import argparse
import os
from enhancement_algorithms import (histogram_equalization,
                                    bilateral_filtering,
                                    total_variation_denoising,
                                    wavelet_denoising)


def enhance(function, origin, destination, technique):
    """

    :param function:
    :param origin:
    :param destination:
    :param technique:
    :return:
    """
    if not os.path.exists(origin):
        print(f"The specified path '{origin}' does not exist.")
        return

    os.makedirs(destination, exist_ok=True)

    for image_name in os.listdir(origin):
        original_image_path = origin + image_name
        enhanced, _ = function(original_image_path)
        enhanced_image_path = destination + technique + image_name
        enhanced.save(enhanced_image_path)


def start_parsing():
    """

    :return:
    """
    parser = argparse.ArgumentParser(
        prog='ImageEnhancer',
        description="This program allows you to enhance the images in a "
                    "directory using one of the many enhancement techniques "
                    "available"
    )

    parser.add_argument(
        '-o', '--origin', required=True, type=str, nargs=1,
        help="Path to directory where the images to be enhanced are located"
    )

    parser.add_argument(
        '-d', '--destination', required=True, type=str, nargs=1,
        help="Path to directory where the enhanced images should be saved in"
    )

    parser.add_argument(
        '-g', '--ground-truth', required=True, type=str, nargs=1,
        help="Path to directory where the ground truth images are located"
    )

    parser.add_argument(
        '-a', '--apply', required=True, type=str, nargs=1,
        help="Specify which enhancement technique to apply to the images:\n"
             "\the (Histogram Equalization)\n"
             "\tbf (Bilateral Filtering)\n"
             "\twv (Wavelet Denoising)\n"
             "\ttv (Total Variation Denoising)\n\n"
             "Alternatively, the metrics for each image in a directory can be "
             "computed with:"
             "\tcm (Compute Metrics)"
    )

    parser.add_argument(
        '-m', '--compute-metrics', required=False, type=str,
    )

    args = parser.parse_args()

    if args.apply == 'he':
        enhance(function=histogram_equalization, origin=args.origin,
                destination=args.destination, technique='he_')
    elif args.apply == 'bf':
        enhance(function=bilateral_filtering, origin=args.origin,
                destination=args.destination, technique='bf_')
    elif args.apply == 'wv':
        enhance(function=wavelet_denoising, origin=args.origin,
                destination=args.destination, technique='wv_')
    elif args.apply == 'tv':
        enhance(function=total_variation_denoising, origin=args.origin,
                destination=args.destination, technique='tv_')
    else:
        print("Invalid arguments!")


if __name__ == '__main__':
    start_parsing()
