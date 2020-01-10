import config as cf
import cv2
import h5py
import json
import numpy as np
import tensorflow as tf
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from utils.hdf5datasetgenerator import HDF5DatasetGenerator
from utils.meanpreprocessor import MeanPreprocessor
from utils.simplepreprocessor import SimplePreprocessor

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def main():
    means = json.loads(open(cf.DATASET_MEAN).read())
    # sp = SimplePreprocessor(cf.WIDTH, cf.HEIGHT)
    # mp = MeanPreprocessor(means["R"], means["G"], means["B"])

    dataset = h5py.File(cf.TRAIN_HDF5, "r")
    x_test = dataset['images'][:]
    y_test = dataset["labels"][:]
    # y_test = tf.keras.utils.to_categorical(y_test, 2)

    # model = tf.keras.models.load_model(cf.MODEL_PATH)
    # result = model.evaluate(x_test, y_test, verbose=1)

    # print(result)

    for i in range(0, 10):
      cv2.imwrite('{}-{}.jpg'.format(i, y_test[i]), x_test[i])

    # y_pred = model.predict(x_test, verbose=1).argmax(axis=1)
    # print(classification_report(y_test, y_pred, target_names=cf.CLASSES))

    # acc = accuracy_score(y_test, y_pred)
    # print("[INFO] score: {}".format(acc))


if __name__ == "__main__":
    main()
