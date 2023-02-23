import argparse
import os.path
from glob import glob
import numpy as np
import cv2

from grid import merge_videos


class DAVIS:
    """
    DAVIS class to encapsulate some information about the dataset
    Arguments:
        root_path: String. Path to the DAVIS dataset.
    """
    def __init__(self, root_path: str):
        self.root_path = os.path.join(root_path)
        self.images_path = os.path.join(self.root_path, "JPEGImages", "480p")
        self.masks_masks = os.path.join(self.root_path, "Annotations_unsupervised", "480p")
        self.images_folders = []
        self.masks_folders = []

    def load_data(self) -> None:
        """
        Loaded all paths for images and masks
        :return: None
        """
        self.images_folders = sorted(glob(os.path.join(self.images_path, "*")))
        self.masks_folders = sorted(glob(os.path.join(self.masks_masks, "*")))


def _pascal_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def overlay_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):
    """ Overlay mask over image.
        This function allows you to overlay a mask over an image with some
        transparency.
        # Arguments
            im: Numpy Array. Array with the image. The shape must be (H, W, 3) and
                the pixels must be represented as `np.uint8` data type.
            ann: Numpy Array. Array with the mask. The shape must be (H, W) and the
                values must be intergers
            alpha: Float. Proportion of alpha to apply at the overlaid mask.
            colors: Numpy Array. Optional custom colormap. It must have shape (N, 3)
                being N the maximum number of colors to represent.
            contour_thickness: Integer. Thickness of each object index contour draw
                over the overlay. This function requires to have installed the
                package `opencv-python`.
        # Returns
            Numpy Array: Image of the overlay with shape (H, W, 3) and data type
                `np.uint8`.
    """
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)

    mask = colors[ann]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(np.uint8),
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE
                                        )[-2:]
            cv2.drawContours(img,
                             contours[0],
                             -1,
                             colors[obj_id].tolist(),
                             contour_thickness)
    return img


def create_one_video(*,
                     video_writer: cv2.VideoWriter,
                     height: int,
                     width: int,
                     path_to_image: str,
                     path_to_masks:str
                     ) -> None:
    """
    Create one video with mask and original image
    :param video_writer: object VideoWriter
    :param height: height images,
    :param width: width images,
    :param path_to_image: path to folder with images
    :param path_to_masks: path to folder with masks
    :return: None
    """
    images_files = sorted(glob(os.path.join(path_to_image, "*.jpg")))
    masks_files = sorted(glob(os.path.join(path_to_masks, "*.png")))

    images = [cv2.imread(file) for file in images_files]
    masks = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in masks_files]

    if len(images) != len(masks_files):
        raise AssertionError("len images and masks must match")

    for i in zip(images, masks):
        image = overlay_mask(im=i[0], ann=i[1])
        resized_img = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        video_writer.write(resized_img)


def create_inline_video(data: DAVIS) -> None:
    """
    Create inline video from images and masks
    :param data: object type DAVIS
    :return: None
    """
    data.load_data()

    output_dir = os.path.join("result_dir")
    os.makedirs(output_dir, exist_ok=True)

    height, width = 480, 854

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(output_dir, "output_line.mp4"), fourcc, 25.0, (width, height))

    for paths in zip(data.images_folders, data.masks_folders):
        create_one_video(video_writer=video_writer,
                         height=height,
                         width=width,
                         path_to_image=paths[0],
                         path_to_masks=paths[1])
        print(f"{paths[0]} ready!")

    video_writer.release()


def create_grid_video(data: DAVIS) -> None:
    """
    Create video in the form of a grid
    :param data: object type DAVIS
    :return: Nobe
    """
    data.load_data()

    output_dir = os.path.join("result_dir")
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    height, width = 480, 854
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer1 = cv2.VideoWriter(os.path.join(tmp_dir, "out1.mp4"), fourcc, 25.0, (width, height))
    video_writer2 = cv2.VideoWriter(os.path.join(tmp_dir, "out2.mp4"), fourcc, 25.0, (width, height))
    video_writer3 = cv2.VideoWriter(os.path.join(tmp_dir, "out3.mp4"), fourcc, 25.0, (width, height))
    video_writer4 = cv2.VideoWriter(os.path.join(tmp_dir, "out4.mp4"), fourcc, 25.0, (width, height))

    count = 0

    for paths in zip(data.images_folders, data.masks_folders):
        if count == 0:
            create_one_video(video_writer=video_writer1,
                             height=height,
                             width=width,
                             path_to_image=paths[0],
                             path_to_masks=paths[1])
            print(f"{paths[0]} ready!")
            count += 1

        elif count == 1:
            create_one_video(video_writer=video_writer2,
                             height=height,
                             width=width,
                             path_to_image=paths[0],
                             path_to_masks=paths[1])
            print(f"{paths[0]} ready!")
            count += 1

        elif count == 2:
            create_one_video(video_writer=video_writer3,
                             height=height,
                             width=width,
                             path_to_image=paths[0],
                             path_to_masks=paths[1])
            print(f"{paths[0]} ready!")
            count += 1

        elif count == 3:
            create_one_video(video_writer=video_writer4,
                             height=height,
                             width=width,
                             path_to_image=paths[0],
                             path_to_masks=paths[1])
            print(f"{paths[0]} ready!")
            count = 0

    video_writer1.release()
    video_writer2.release()
    video_writer3.release()
    video_writer4.release()

    videos_to_merge = [
        os.path.join(tmp_dir, 'out1.mp4'),
        os.path.join(tmp_dir, 'out2.mp4'),
        os.path.join(tmp_dir, 'out3.mp4'),
        os.path.join(tmp_dir, 'out4.mp4')
    ]
    merge_videos(
        videos_to_merge,
        os.path.join(output_dir, 'output_grid.mp4'),
        grid_size=(2, 2),)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_to_dataset", type=str)
    parser.add_argument("-g", "--grid", default=False, type=bool)
    args = parser.parse_args()

    dataset = DAVIS(root_path=args.path_to_dataset)
    if args.grid:
        create_grid_video(dataset)
    else:
        create_inline_video(data=dataset)

