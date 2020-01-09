import numpy as np
import pickle
import random
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD

WIDTH = 224
HEIGHT = 224
CHANNEL = 3
CLASSES = ['hand', 'non-hand']

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def train():
    le = LabelBinarizer()
    le.fit(CLASSES)

    with open('train1.pickle', 'rb') as f:
        train = pickle.load(f)
    random.shuffle(train)
    x_train = np.zeros((len(train), WIDTH, HEIGHT, CHANNEL))
    y_train = np.zeros((len(train), 1))
    for i, sample in enumerate(train):
        x_train[i] = sample['image']
        y_train[i] = le.transform([sample['label']])
    x_train = x_train / 255.0
    y_train = np.hstack([y_train, 1 - y_train])

    with open('test1.pickle', 'rb') as f:
        test = pickle.load(f)
    random.shuffle(test)
    x_test = np.zeros((len(test), WIDTH, HEIGHT, CHANNEL))
    y_test = np.zeros((len(test), 1))
    for i, sample in enumerate(test):
        x_test[i] = sample['image']
        y_test[i] = le.transform([sample['label']])
    x_test = x_test / 255.0
    y_test = np.hstack([y_test, 1 - y_test])

    epochs_num = 100
    opt = SGD(lr=0.01, decay=0.01/epochs_num)
    model = MobileNetV2(
        include_top=True, weights=None, input_shape=(WIDTH, HEIGHT, CHANNEL), classes=len(CLASSES))
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=16, epochs=epochs_num, verbose=1)

    # preds = model.predict(x_test)
    # print(preds)
    # result = classification_report(y_test[:, 0], preds[:, 0], target_names=CLASSES)
    # print(result)

    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()
