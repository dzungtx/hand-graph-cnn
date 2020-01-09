import cv2
import numpy as np
import os
import pickle

from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

WIDTH = 224
HEIGHT = 224
CHANNEL = 3


def extract():
    dataset = []
    dataset += extract_images_in_folder(
        '/home/dzung/SSD500G/data/ouhands/train/train/hand_data/colour/', 'hand')
    dataset += extract_images_in_folder(
        '/home/dzung/SSD500G/data/ouhands/train/train/negative_data/colour/', 'non-hand')
    with open('train.pickle', 'wb') as f:
        pickle.dump(dataset, f)

    dataset = []
    dataset += extract_images_in_folder(
        '/home/dzung/SSD500G/data/ouhands/test/hand_data/colour/', 'hand')
    dataset += extract_images_in_folder(
        '/home/dzung/SSD500G/data/ouhands/test/negative_data/colour/', 'non-hand')
    with open('test.pickle', 'wb') as f:
        pickle.dump(dataset, f)


def extract_images_in_folder(folder, label):
    dataset = []
    file_names = os.listdir(folder)
    count = 0
    for file_name in file_names:
        image = load_img(os.path.join(folder, file_name),
                         target_size=(WIDTH, HEIGHT))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        dataset.append({'image': image, 'label': label})
        count += 1
        if count % 100 == 0:
            print("{}/{}".format(count, len(file_names)), end='\r')
    return dataset


if __name__ == "__main__":
    extract()
