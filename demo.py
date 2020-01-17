# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Basic evaluation script for PyTorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import numpy as np
import os.path as osp
import time
import torch

from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network import ShapePoseNetwork
from hand_shape_pose.data.build import build_dataset

from hand_shape_pose.util.logger import setup_logger, get_logger_filename
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis import draw_2d_skeleton, draw_3d_skeleton
from hand_shape_pose.util import renderer


def main():
    parser = argparse.ArgumentParser(
        description="3D Hand Shape and Pose Inference")
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
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.load_model(cfg)

    mesh_renderer = renderer.MeshRenderer(model.hand_tri.astype('uint32'))

    # 3. Inference
    model.eval()
    cpu_device = torch.device("cpu")

    cam_params = torch.from_numpy(
        np.array([123, 123, 10, 10])).unsqueeze(0).to(cpu_device)

    bboxes = torch.from_numpy(
        np.array([80., 64., 493., 493.])).unsqueeze(0).to(cpu_device)

    pose_roots = torch.from_numpy(
        np.array([0.41856557, 11.017581, 42.429573])).unsqueeze(0).to(cpu_device)

    pose_scales = (torch.ones((1,)) * 2.0).to(cpu_device)

    img = cv2.imread('data/random/images/1.jpg')
    images = torch.from_numpy(img).unsqueeze(0).to(cpu_device)

    with torch.no_grad():
        est_mesh_cam_xyz, est_pose_uv, est_pose_cam_xyz = \
            model(images, cam_params, bboxes, pose_roots, pose_scales)
        est_mesh_cam_xyz = [o.to(cpu_device) for o in est_mesh_cam_xyz]
        est_pose_uv = [o.to(cpu_device) for o in est_pose_uv]
        est_pose_cam_xyz = [o.to(cpu_device) for o in est_pose_cam_xyz]

    print(est_pose_cam_xyz[0])

    if cfg.EVAL.SAVE_BATCH_IMAGES_PRED:
        file_name = 'output/configs/eval_random.yaml/full_1.jpg'
        print("Saving image: {}".format(file_name))
        save_image(mesh_renderer, images.to(cpu_device), cam_params.to(cpu_device),
                   bboxes.to(cpu_device), est_mesh_cam_xyz, est_pose_uv, est_pose_cam_xyz, file_name)


def save_image(mesh_renderer, batch_images, cam_params, bboxes,
               est_mesh_cam_xyz, est_pose_uv, est_pose_cam_xyz,
               file_name, padding=2):
    num_images = batch_images.shape[0]
    image_height = batch_images.shape[1]
    image_width = batch_images.shape[2]
    num_column = 4

    grid_image = np.zeros((num_images * (image_height + padding), num_column * (image_width + padding), 3),
                          dtype=np.uint8)

    for id_image in range(num_images):
        image = batch_images[id_image].numpy()
        cam_param = cam_params[id_image].numpy()
        box = bboxes[id_image].numpy()
        mesh_xyz = est_mesh_cam_xyz[id_image].numpy()
        pose_uv = est_pose_uv[id_image].numpy()
        pose_xyz = est_pose_cam_xyz[id_image].numpy()

        rend_img_overlay = draw_mesh(
            mesh_renderer, image, cam_param, box, mesh_xyz)
        skeleton_overlay = draw_2d_skeleton(image, pose_uv)
        skeleton_3d = draw_3d_skeleton(pose_xyz, image.shape[:2])

        img_list = [image, rend_img_overlay, skeleton_overlay, skeleton_3d]

        height_begin = (image_height + padding) * id_image
        height_end = height_begin + image_height
        width_begin = 0
        width_end = image_width
        for show_img in img_list:
            grid_image[height_begin:height_end,
                       width_begin:width_end, :] = show_img[..., :3]
            width_begin += (image_width + padding)
            width_end = width_begin + image_width

    cv2.imwrite(file_name, grid_image)


def draw_mesh(mesh_renderer, image, cam_param, box, mesh_xyz):
    resize_ratio = float(image.shape[0]) / box[2]
    cam_for_render = np.array(
        [cam_param[0], cam_param[2] - box[0], cam_param[3] - box[1]]) * resize_ratio

    rend_img_overlay = mesh_renderer(
        mesh_xyz, cam=cam_for_render, img=image, do_alpha=True)

    return rend_img_overlay


if __name__ == "__main__":
    main()
