import argparse

def start_parsing():
    parser = argparse.ArgumentParser(
        prog='ImageEnhancer',
        description="This program allows you to enhance the images in a "
                    "directory using one of the many enhancement techniques "
                    "available"
    )

    parser.add_argument(
        '-s', '--source', required=True, type=str, nargs=1,
        help="Path to the directory where the images to be enhanced are located"
    )

    parser.add_argument(
        '-d', '--destination', required=True, type=str, nargs=1,
        help="Path to the directory where the enhanced images should be saved in"
    )

    parser.add_argument(
        '-t', '--technique', required=True, type=str, nargs=1,
        help="Specify which enhancement technique to apply to the images.\n"
             "he (Histrogram Equalization)\n"
             "bf (Bilateral Filtering)\n"
             "wv (Wavelet Denoising)\n"
             "tv (Total Variation Denoising)\n"
    )

if __name__ == '__main__':
    pass
