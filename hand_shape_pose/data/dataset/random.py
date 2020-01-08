# Copyright (c) Liuhao Ge. All Rights Reserved.
"""
Real world test set
"""
import os
import os.path as osp
import logging
import cv2
import numpy as np

import torch
import torch.utils.data


class Random(torch.utils.data.Dataset):
    def __init__(self, root):
        self.data_path = osp.join(root, 'images')

        self.image_names = os.listdir(self.data_path)

        self.cam_params = torch.from_numpy(
            np.array([923.648, 923.759, 640.047, 362.336]))

        self.bboxes = torch.from_numpy(np.array([380., 164., 493., 493.]))

        self.pose_roots = torch.from_numpy(
            np.array([0.41856557, 11.017581, 42.429573]))

        self.pose_scales = 5.0

    def __getitem__(self, index):
        img = cv2.imread(
            osp.join(self.data_path, self.image_names[index]))
        img = torch.from_numpy(img)  # 256 x 256 x 3
        return img, self.cam_params, self.bboxes, self.pose_roots, self.pose_scales, index, self.image_names[index]

    def __len__(self):
        return len(self.image_names)
