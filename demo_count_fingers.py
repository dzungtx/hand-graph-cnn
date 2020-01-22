import argparse
import cv2
import math
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

CONFIG_FILE = 'configs/eval_random.yaml'
WIDTH = 256
HEIGHT = 256
WINDOW_SIZE = 7


def main():
    global device

    parser = argparse.ArgumentParser(description="Hand gesture recognition")
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

    fingers_num_series = []

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

        n = count_fingers(hand_pose) if hand_pose is not None else None
        fingers_num_series.append(n)
        if len(fingers_num_series) > WINDOW_SIZE:
            fingers_num_series.pop(0)

        if len(fingers_num_series) == WINDOW_SIZE and fingers_num_series.count(None) < WINDOW_SIZE // 3:
            fingers_num = int(
                np.median([i for i in fingers_num_series if i is not None]))
        else:
            fingers_num = 0

        t2 = time.time()
        fps = round(1 / (t2 - t1), 1)
        frame = visualize_hand_pose(
            frame, hand_pose, fingers_num=fingers_num, fps=fps)

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
        est_pose_uv = model.cacl_2d_pose(
            torch.from_numpy(image).unsqueeze(0).to(device))
    return est_pose_uv[0].cpu().numpy()


def validate_hand_pose(hand_pose):
    confidence = np.median(hand_pose[:, 2])
    return confidence > 0.8


def count_fingers(hand_pose):
    count = 0
    p0 = hand_pose[0, :]
    for i in range(5):
        p1 = hand_pose[4 * i + 1]
        p2 = hand_pose[4 * i + 2]
        p3 = hand_pose[4 * i + 3]
        p4 = hand_pose[4 * i + 4]
        if min(p1[2], p2[2], p3[2], p4[2]) < 0.6:
            continue
        d01 = calc_distance(p0, p1)
        d12 = calc_distance(p1, p2)
        d23 = calc_distance(p2, p3)
        d34 = calc_distance(p3, p4)
        d04 = calc_distance(p0, p4)
        if min(d01, d12, d23, d34) > 10 and d01 + d12 + d23 + d34 < d04 * 1.05:
            count += 1
    return count


def calc_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def visualize_hand_pose(image, hand_pose, fingers_num=None, fps=None):
    if hand_pose is not None:
        image = draw_2d_skeleton(image, hand_pose)
    if fingers_num is not None:
        cv2.putText(image, "{}".format(fingers_num), (20, 30),
                    0, 3e-3 * HEIGHT, (0, 255, 0), thickness=2)
    if fps is not None:
        cv2.putText(image, "{} fps".format(fps), (WIDTH - 80, HEIGHT - 12),
                    0, 2.5e-3 * HEIGHT, (0, 255, 0), thickness=2)
    return image


if __name__ == "__main__":
    main()
