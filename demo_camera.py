import argparse
import cv2
import numpy as np
import os.path as osp
import time
import torch

from gracefully_killer import GracefulKiller
from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network import ShapePoseNetwork
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis import draw_2d_skeleton
from scipy.sparse.linalg.interface import LinearOperator

WIDTH = 256
HEIGHT = 256
CONFIG_FILE = 'configs/eval_random.yaml'


def main():
    global device

    parser = argparse.ArgumentParser(description="Hand gesture recognization")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cfg.merge_from_file(CONFIG_FILE)
    cfg.freeze()
    device = torch.device(cfg.MODEL.DEVICE)

    hand_shape_pose_model = init_hand_shape_pose_model(cfg)

    cam = cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    killer = GracefulKiller()

    while not killer.kill_now:
        t1 = time.time()

        if cv2.waitKey(1) % 256 == 27:  # Press ESC
            break

        ret, frame = cam.read()
        if not ret:
            print('Can not capture from camera')
            break

        frame = cv2.resize(frame, (HEIGHT, WIDTH))

        hand_pose = extract_hand_pose(hand_shape_pose_model, frame)
        if not validate_hand_pose(hand_pose):
            hand_pose = None

        t2 = time.time()
        fps = round(1 / (t2 - t1), 1)
        frame = visualize_hand_pose(frame, hand_pose, fps=fps)

        cv2.imshow('Hand pose detection', frame)

    cv2.destroyAllWindows()
    cam.release()


def init_hand_shape_pose_model(cfg):
    output_dir = osp.join(cfg.EVAL.SAVE_DIR, CONFIG_FILE)
    mkdir(output_dir)
    model = ShapePoseNetwork(cfg, output_dir)
    model.to(device)
    model.load_model(cfg)
    model.eval()
    return model


def extract_hand_pose(model, image):
    with torch.no_grad():
        est_pose_uv = model.cacl_pose(
            torch.from_numpy(image).unsqueeze(0).to(device))
    return est_pose_uv[0].cpu().numpy()


def validate_hand_pose(hand_pose):
    confidence = np.median(hand_pose[:, 2])
    return confidence > 0.8


def visualize_hand_pose(image, hand_pose, fps=None):
    if hand_pose is not None:
        image = draw_2d_skeleton(image, hand_pose)
    if fps is not None:
        cv2.putText(image, "{} fps".format(fps), (WIDTH - 80, HEIGHT - 12),
                    0, 2.5e-3 * HEIGHT, (0, 255, 0), thickness=2)
    return image


if __name__ == "__main__":
    main()
