import argparse
import cv2
import numpy as np
import os.path as osp
import time
import torch

from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network import ShapePoseNetwork
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis import draw_2d_skeleton
from scipy.sparse.linalg.interface import LinearOperator


def main():
    parser = argparse.ArgumentParser(description="Hand gesture recognization")
    parser.add_argument(
        "--config-file",
        default="configs/eval_random.yaml",
        metavar="FILE",
        help="path to config file",
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_dir = osp.join(cfg.EVAL.SAVE_DIR, args.config_file)
    mkdir(output_dir)

    # 1. Load network model
    model = ShapePoseNetwork(cfg, output_dir)
    model.load_model(cfg)

    # 2. Inference
    model.eval()

    img = cv2.imread('data/random/images/1.jpg')

    with torch.no_grad():
        est_pose_uv = model.cacl_2d_pose(torch.from_numpy(img).unsqueeze(0))

    pose_uv = est_pose_uv[0].numpy()

    save_image(img, pose_uv,
               'output/configs/eval_random.yaml/1.jpg')


def save_image(image, pose_uv, file_name, padding=2):
    print("Saving image: {}".format(file_name))

    image_height = image.shape[0]
    image_width = image.shape[1]
    num_column = 2

    grid_image = np.zeros((image_height + padding, num_column * (image_width + padding), 3),
                          dtype=np.uint8)

    skeleton_overlay = draw_2d_skeleton(image, pose_uv)

    img_list = [image, skeleton_overlay]

    width_begin = 0
    width_end = image_width
    for show_img in img_list:
        grid_image[:image_height,
                    width_begin:width_end, :] = show_img[..., :3]
        width_begin += (image_width + padding)
        width_end = width_begin + image_width

    cv2.imwrite(file_name, grid_image)


if __name__ == "__main__":
    main()
