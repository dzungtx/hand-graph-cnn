import argparse
import cv2
import numpy as np
import os.path as osp
import time
import torch
import tensorflow as tf

from PIL import Image
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

    hand_shape_pose_model = init_hand_shape_pose_model(args.config_file)
    hand_classifier_model = init_hand_classifier_model(
        'hand_classifier/model_output/mobilenetv2.tflite')

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cam.read()
        if not ret:
            print('Can not capture from camera')
            break

        cv2.imshow("Webcam", frame)

        if classify_image(hand_classifier_model, frame) != 0:
            cv2.imshow('Gesture Recognition', frame)
            continue

        hand_pose = extract_hand_pose(hand_shape_pose_model, frame)
        cv2.imshow('Gesture Recognition', hand_pose)

        k = cv2.waitKey(1)
        if k % 256 == 27: # ESC
            break

    cv2.destroyAllWindows()
    cam.release()


def init_hand_shape_pose_model(config_file):
    cfg.merge_from_file(config_file)
    cfg.freeze()
    output_dir = osp.join(cfg.EVAL.SAVE_DIR, config_file)
    mkdir(output_dir)
    model = ShapePoseNetwork(cfg, output_dir)
    model.load_model(cfg)
    model.eval()
    return model

def extract_hand_pose(model, image):
    img = cv2.resize(image, (256, 256))
    with torch.no_grad():
        est_pose_uv = model(torch.from_numpy(img).unsqueeze(0))
    pose_uv = est_pose_uv[0].numpy()
    skeleton_overlay = draw_2d_skeleton(img, pose_uv)
    return skeleton_overlay


def init_hand_classifier_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def classify_image(interpreter, image):
    input_mean = 127.5
    input_std = 127.5
    height = 224
    width = 224

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    img = cv2.resize(image, (height, width))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    return results.argsort()[-1:][::-1][0]


if __name__ == "__main__":
    main()
