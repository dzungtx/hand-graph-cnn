import config as cf
import numpy as np
import json
import pickle
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto

from model_builder.mobilenet_v3_large import MobileNetV3
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.hdf5datasetgenerator import HDF5DatasetGenerator
from utils.meanpreprocessor import MeanPreprocessor
from utils.patchpreprocessor import PatchPreprocessor
from utils.simplepreprocessor import SimplePreprocessor

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def train():
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                             width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15,
                             horizontal_flip=True, fill_mode="nearest")

    means = json.loads(open(cf.DATASET_MEAN).read())

    sp = SimplePreprocessor(cf.WIDTH, cf.HEIGHT)
    pp = PatchPreprocessor(cf.WIDTH, cf.HEIGHT)
    mp = MeanPreprocessor(means["R"], means["G"], means["B"])

    train_gen = HDF5DatasetGenerator(cf.TRAIN_HDF5, cf.BATCH_SIZE, classes=cf.CLASSES_N, aug=aug)
    val_gen = HDF5DatasetGenerator(cf.VAL_HDF5, cf.BATCH_SIZE, classes=cf.CLASSES_N)

    epochs_num = 100
    # opt = SGD(lr=0.1, momentum=0.9, decay=0.1/epochs_num)
    opt = Adam(lr=1e-3)
    model = MobileNetV3(num_classes=cf.CLASSES_N, l2_reg=1e-2)
    # model = VGG16(include_top=True, weights=None, input_shape=(cf.WIDTH, cf.HEIGHT, cf.CHANNEL), classes=cf.CLASSES_N)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    model.fit(train_gen.generator(),
              steps_per_epoch=train_gen.numImages // cf.BATCH_SIZE // 10,
              validation_data=val_gen.generator(),
              validation_steps=val_gen.numImages // cf.BATCH_SIZE,
              epochs=epochs_num,
              verbose=1)

    model.save(cf.MODEL_PATH)

    train_gen.close()
    val_gen.close()


if __name__ == "__main__":
    train()
