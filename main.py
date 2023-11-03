"""
    describe the program
"""

from pathlib import Path
import stats
import argparse
import os
import cv2 as cv
from enhancement_algorithms import (histogram_equalization,
                                    bilateral_filtering,
                                    total_variation_denoising,
                                    wavelet_denoising,
                                    clahe)


def enhance(function, input_path: Path, output_path: Path,
            ground_truth_path: Path, technique_prefix: str):
    if not ground_truth_path.exists():
        print(f"The specified path '{ground_truth_path}' does not exist.")
        return

    if not input_path.exists():
        print(f"The specified path '{input_path}' does not exist.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    output_csv_path = output_path / "csv"
    output_csv_path.mkdir(exist_ok=True)

    output_metrics = []

    for original_image_path in input_path.glob("*"):
        image_name = original_image_path.name
        original_image_array = cv.imread(str(original_image_path),
                                         cv.IMREAD_GRAYSCALE)

        enhanced_image_array, runtime = function(original_image_array)

        output_image_path = output_path / (technique_prefix + image_name)
        cv.imwrite(str(output_image_path), enhanced_image_array)

        image_name_stem = os.path.splitext(image_name)[0]

        ground_truth_image_path = ground_truth_path / (image_name_stem +
                                                       '_mask.png')
        ground_truth_image_array = cv.imread(str(ground_truth_image_path),
                                             cv.IMREAD_GRAYSCALE)

        output_metrics.append(
            {'image': technique_prefix + image_name,
             'ssim': stats.compute_ssim(ground_truth_image_array,
                                        enhanced_image_array),
             'cnr': stats.compute_cnr(enhanced_image_array,
                                      ground_truth_image_array),
             'ambe': stats.compute_ambe(original_image_array,
                                        enhanced_image_array),
             'runtime': runtime,
             'mse': stats.compute_mse(original_image_array,
                                      enhanced_image_array),
             'rmse': stats.compute_rmse(original_image_array,
                                        enhanced_image_array),
             'psnr': stats.compute_psnr(original_image_array,
                                        enhanced_image_array),
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
             "\tclahe (Contrast Limited Adaptative Histogram Equalization"
    )

    parser.add_argument(
        '-m', '--compute-metrics', required=False, type=str,
    )

    args = parser.parse_args()

    if 'he' in args.apply:
        enhance(function=histogram_equalization,
                input_path=Path(args.origin[0]),
                output_path=Path(args.destination[0]),
                technique_prefix='he_',
                ground_truth_path=Path(args.ground_truth[0]))
    elif 'bf' in args.apply:
        enhance(function=bilateral_filtering,
                input_path=Path(args.origin[0]),
                output_path=Path(args.destination[0]),
                technique_prefix='bf_',
                ground_truth_path=Path(args.ground_truth[0]))
    elif 'wv' in args.apply:
        enhance(function=wavelet_denoising,
                input_path=Path(args.origin[0]),
                output_path=Path(args.destination[0]),
                technique_prefix='wv_',
                ground_truth_path=Path(args.ground_truth[0]))
    elif 'tv' in args.apply:
        enhance(function=total_variation_denoising,
                input_path=Path(args.origin[0]),
                output_path=Path(args.destination[0]),
                technique_prefix='tv_',
                ground_truth_path=Path(args.ground_truth[0]))
    elif 'clahe' in args.apply:
        enhance(function=clahe,
                input_path=Path(args.origin[0]),
                output_path=Path(args.destination[0]),
                technique_prefix='clahe_',
                ground_truth_path=Path(args.ground_truth[0]))
    else:
        print(args.apply)
        print("Invalid arguments!")


if __name__ == '__main__':
    start_parsing()
