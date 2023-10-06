"""
    Enhancement algorithms
"""

from PIL import Image


def get_grayscale_image(image_path):
    """

    :param image_path:
    :return:
    """
    try:
        with Image.open(image_path) as image:
            if image.mode == 'L':
                grayscale_image = image.copy()
            else:
                grayscale_image = image.convert('L')
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        grayscale_image = None
    return grayscale_image
