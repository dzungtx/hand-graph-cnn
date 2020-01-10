import config as cf
import cv2
import json
import numpy as np
import os
import progressbar
import random

from utils.aspectawarepreprocessor import AspectAwarePreprocessor
from utils.simplepreprocessor import SimplePreprocessor
from utils.hdf5datasetwriter import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder


def main():
    w = int(cf.WIDTH * 1)
    h = int(cf.HEIGHT * 1)

    datasets = [("train", cf.TRAIN_POSITIVES, cf.TRAIN_NEGATIVES, cf.TRAIN_HDF5),
                ("val", cf.VAL_POSITIVES, cf.VAL_NEGATIVES, cf.VAL_HDF5)]

    aap = SimplePreprocessor(w, h)
    (R, G, B) = ([], [], [])

    le = LabelEncoder()
    le.fit(cf.CLASSES)

    for (dtype, pos_input, neg_input, output) in datasets:
        print("[INFO] Building {}".format(output))
        if os.path.isfile(output):
            os.remove(output)

        pos_image_paths = [os.path.join(pos_input, iname) for iname in os.listdir(pos_input)]
        neg_image_paths = [os.path.join(neg_input, iname)
                        for iname in os.listdir(neg_input)]
        image_paths = pos_image_paths + neg_image_paths
        random.shuffle(image_paths)

        labels = [p.split(os.path.sep)[-3].split('_')[0] for p in image_paths]
        labels = le.transform(labels)

        dataset = HDF5DatasetWriter(
            (len(image_paths), w, h, cf.CHANNEL), output, dataKey='images')
        dataset.storeClassLabels(le.classes_)

        widgets = ["Processing images: ", progressbar.Percentage(), " ",
                progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(
            maxval=len(image_paths), widgets=widgets).start()

        for (image_path, label) in zip(image_paths, labels):
            image = cv2.imread(image_path)
            image = aap.preprocess(image)

            if dtype == "train":
                (b, g, r) = cv2.mean(image)[:3]
                R.append(r)
                G.append(g)
                B.append(b)

            dataset.add([image], [label])
            pbar.update(pbar.currval + 1)

        dataset.close()
        pbar.finish()

    print("[INFO] Serializing means")
    D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
    with open(cf.DATASET_MEAN, "w") as f:
        f.write(json.dumps(D))


if __name__ == "__main__":
    main()
