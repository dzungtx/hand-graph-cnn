import config as cf
import numpy as np
import json
import pickle
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto

from utils.hdf5datasetgenerator import HDF5DatasetGenerator
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.simplepreprocessor import SimplePreprocessor
from utils.patchpreprocessor import PatchPreprocessor
from utils.meanpreprocessor import MeanPreprocessor

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def train():
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
                             width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                             horizontal_flip=True, fill_mode="nearest")

    means = json.loads(open(cf.DATASET_MEAN).read())

    sp = SimplePreprocessor(cf.WIDTH, cf.HEIGHT)
    pp = PatchPreprocessor(cf.WIDTH, cf.HEIGHT)
    mp = MeanPreprocessor(means["R"], means["G"], means["B"])

    train_gen = HDF5DatasetGenerator(cf.TRAIN_HDF5, cf.BATCH_SIZE, classes=cf.CLASSES_N)
    val_gen = HDF5DatasetGenerator(cf.VAL_HDF5, cf.BATCH_SIZE, classes=cf.CLASSES_N)

    epochs_num = 1
    # opt = SGD(lr=0.01, momentum=0.9, decay=0.01/epochs_num)
    opt = Adam(lr=1e-3)
    model = MobileNetV2(
        include_top=True, weights=None, input_shape=(cf.WIDTH, cf.HEIGHT, cf.CHANNEL), classes=cf.CLASSES_N)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    model.fit(train_gen.generator(),
              steps_per_epoch=train_gen.numImages // cf.BATCH_SIZE,
              validation_data=val_gen.generator(),
              validation_steps=val_gen.numImages // cf.BATCH_SIZE,
              epochs=epochs_num,
              max_queue_size=10,
              callbacks=[], verbose=1)

    model.save(cf.MODEL_PATH)

    train_gen.close()
    val_gen.close()


if __name__ == "__main__":
    train()
