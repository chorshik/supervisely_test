import argparse
import os
import cv2
import numpy as np


def merge_images(*, directory_path: str) -> np.ndarray:
    """
    Combine the image from the sliced pictures and from them collects the original image
    :param directory_path:
    :return: ndarray
    """
    name, data = os.path.basename(os.path.normpath(directory_path)).split("output")
    data = data.split("_")[1:]
    name = name.rstrip("_").rstrip('_')

    height = int(data[0])
    width = int(data[1])
    shape = int(data[2])
    window_width, window_height = map(int, data[3].split("x"))
    y_offset = int(data[4][1:])
    x_offset = int(data[5][1:])

    output_dir = os.path.join("result_merge")
    os.makedirs(output_dir, exist_ok=True)

    merged_image = np.zeros((height, width, shape), np.uint8)

    for y in range(0, height, y_offset):
        for x in range(0, width, x_offset):
            window_name = f"{y}_{x}"

            window_path = os.path.join(directory_path, window_name + ".png")
            window = cv2.imread(window_path)

            merged_image[y:y + window_height, x:x + window_width] = window

    cv2.imwrite(os.path.join(output_dir, f"{name}_merged.jpg"), merged_image)

    return merged_image


def verify_image(*, image_path: str, merged_image: np.ndarray) -> bool:
    """
    Verify the merged image with the original
    :param image_path: path to original image
    :param merged_image: merged image as ndarray
    :return: bool
    """
    original_image = cv2.imread(image_path)

    return np.array_equal(original_image, merged_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_to_folder", type=str)
    parser.add_argument("-o", "--path_to_image", type=str)
    args = parser.parse_args()

    image = merge_images(directory_path=args.path_to_folder)

    if verify_image(image_path=args.path_to_image,
                    merged_image=image):
        print("OK")
    else:
        print("Merged image has different")
