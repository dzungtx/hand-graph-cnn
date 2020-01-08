# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Basic evaluation script for PyTorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os.path as osp
import time
import torch

from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network import ShapePoseNetwork
from hand_shape_pose.data.build import build_dataset

from hand_shape_pose.util.logger import setup_logger, get_logger_filename
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis import save_batch_image_with_mesh_joints
from hand_shape_pose.util import renderer


def main():
    parser = argparse.ArgumentParser(
        description="3D Hand Shape and Pose Inference")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = osp.join(cfg.EVAL.SAVE_DIR, args.config_file)
    mkdir(output_dir)

    # 1. Load network model
    model = ShapePoseNetwork(cfg, output_dir)
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.load_model(cfg)

    mesh_renderer = renderer.MeshRenderer(model.hand_tri.astype('uint32'))

    # 2. Load data
    dataset_val = build_dataset(cfg.EVAL.DATASET)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.MODEL.BATCH_SIZE,
        num_workers=cfg.MODEL.NUM_WORKERS
    )

    # 3. Inference
    model.eval()
    results_pose_cam_xyz = {}
    cpu_device = torch.device("cpu")

    for i, batch in enumerate(data_loader_val):
        t1 = time.time()

        images, cam_params, bboxes, pose_roots, pose_scales, image_ids, image_names = batch

        images, cam_params, bboxes, pose_roots, pose_scales = \
            images.to(device), cam_params.to(device), bboxes.to(
                device), pose_roots.to(device), pose_scales.to(device)
        with torch.no_grad():
            est_mesh_cam_xyz, est_pose_uv, est_pose_cam_xyz = \
                model(images, cam_params, bboxes, pose_roots, pose_scales)
            est_mesh_cam_xyz = [o.to(cpu_device) for o in est_mesh_cam_xyz]
            est_pose_uv = [o.to(cpu_device) for o in est_pose_uv]
            est_pose_cam_xyz = [o.to(cpu_device) for o in est_pose_cam_xyz]

        t2 = time.time()
        print(round(t2 - t1, 2))

        results_pose_cam_xyz.update(
            {img_id.item(): result for img_id, result in zip(image_ids, est_pose_cam_xyz)})

        if cfg.EVAL.SAVE_BATCH_IMAGES_PRED:
            file_name = '{}_{}.jpg'.format(
                osp.join(output_dir, 'pred'), '-'.join(['.'.join(name.split('.')[:-1]) for name in image_names]))
            print("Saving image: {}".format(file_name))
            save_batch_image_with_mesh_joints(mesh_renderer, images.to(cpu_device), cam_params.to(cpu_device),
                                              bboxes.to(
                                                  cpu_device), est_mesh_cam_xyz, est_pose_uv,
                                              est_pose_cam_xyz, file_name)


if __name__ == "__main__":
    main()
