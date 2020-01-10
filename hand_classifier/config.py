WIDTH = 224
HEIGHT = 224
CHANNEL = 3

CLASSES = ['negative', 'hand']
CLASSES_N = 2

TRAIN_POSITIVES = '/home/dzung/SSD500G/data/ouhands/train/train/hand_data/colour/'
TRAIN_NEGATIVES = '/home/dzung/SSD500G/data/ouhands/train/train/negative_data/colour/'

VAL_POSITIVES = '/home/dzung/SSD500G/data/ouhands/test/hand_data/colour/'
VAL_NEGATIVES = '/home/dzung/SSD500G/data/ouhands/test/negative_data/colour/'

TRAIN_HDF5 = 'train.hdf5'
VAL_HDF5 = 'val.hdf5'

DATASET_MEAN = 'mean.json'

BATCH_SIZE = 48
MODEL_PATH = 'model'