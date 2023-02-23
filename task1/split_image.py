import argparse
import os
import cv2


def split_image(*, image_path: str, window_size: str, x_offset: int, y_offset: int) -> None:
    """
    Slices images with a sliding window approach.
    :param image_path: path to original image
    :param window_size: sliced image size
    :param x_offset: offset in x
    :param y_offset: offset in y
    :return: None
    """
    image = cv2.imread(image_path)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    height, width, shape = image.shape

    window_width, window_height = map(int, window_size.split("x"))

    output_dir = os.path.join("result_split", f"{image_name}_output_"
                                              f"{height}_{width}_{shape}_{window_width}x{window_height}_"
                                              f"x{x_offset}_y{y_offset}")
    os.makedirs(output_dir, exist_ok=True)

    for y in range(0, height, y_offset):
        for x in range(0, width, x_offset):
            window = image[y:y + window_height, x:x + window_width]
            window_name = f"{y}_{x}"
            cv2.imwrite(os.path.join(output_dir, window_name + ".png"), window)

    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-s", "--size", type=str)
    parser.add_argument("-x", "--x", type=int)
    parser.add_argument("-y", "--y", type=int)
    args = parser.parse_args()

    split_image(image_path=args.path,
                window_size=args.size,
                x_offset=args.x,
                y_offset=args.y)
