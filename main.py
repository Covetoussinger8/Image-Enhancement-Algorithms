"""
    describe the program
"""

import stats
import argparse
import os
from enhancement_algorithms import (histogram_equalization,
                                    bilateral_filtering,
                                    total_variation_denoising,
                                    wavelet_denoising)


def enhance(function, origin, destination, ground_truth_path, technique):
    """

    :param function:
    :param origin:
    :param destination:
    :param technique:
    :return:
    """
    if not os.path.exists(ground_truth_path):
        print(f"The specified path '{origin}' does not exist.")
        return

    if not os.path.exists(origin):
        print(f"The specified path '{origin}' does not exist.")
        return

    os.makedirs(destination, exist_ok=True)
    output_csv_path = destination + "csv/"
    os.makedirs(destination, exist_ok=True)

    output_metrics = []

    for image_name in os.listdir(origin):
        input_image_path = origin + image_name
        enhanced, runtime = function(input_image_path)
        output_image_path = destination + technique + image_name
        enhanced.save(output_image_path)
        image_name_stem = os.path.splitext(image_name)[0]
        ground_truth_image_path = (ground_truth_path + image_name_stem +
                                   '_mask.png')
        output_metrics.append(
            {"image": image_name,
             "mse": stats.compute_mse(input_image_path, output_image_path),
             "rmse": stats.compute_rmse(input_image_path, output_image_path),
             "cnr": stats.compute_cnr(output_image_path,
                                      ground_truth_image_path),
             "ambe": stats.compute_ambe(input_image_path, output_image_path),
             "psnr": stats.compute_psnr(input_image_path, output_image_path),
             "ssim": stats.compute_ssim(input_image_path, output_image_path),
             "runtime": runtime
             }
        )

    stats.generate_csv(output_csv_path, output_metrics)


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
        '-g', '--ground_truth', required=True, type=str, nargs=1,
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

    if 'he' in args.apply:
        enhance(function=histogram_equalization, origin=args.origin[0],
                destination=args.destination[0], technique='he_',
                ground_truth_path=args.ground_truth[0])
    elif 'bf' in args.apply:
        enhance(function=bilateral_filtering, origin=args.origin[0],
                destination=args.destination[0], technique='bf_',
                ground_truth_path=args.ground_truth[0])
    elif 'wv' in args.apply:
        enhance(function=wavelet_denoising, origin=args.origin[0],
                destination=args.destination[0], technique='wv_',
                ground_truth_path=args.ground_truth[0])
    elif 'tv' in args.apply:
        enhance(function=total_variation_denoising, origin=args.origin[0],
                destination=args.destination[0], technique='tv_',
                ground_truth_path=args.ground_truth[0])
    else:
        print(args.apply)
        print("Invalid arguments!")


if __name__ == '__main__':
    start_parsing()
