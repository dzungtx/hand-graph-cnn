import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.aspectawarepreprocessor import AspectAwarePreprocessor


def main():
    aap = AspectAwarePreprocessor(256, 256)
    img = cv2.imread('landscape.jpg')
    img = aap.preprocess(img)

    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
                             width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                             horizontal_flip=True, fill_mode="nearest")
    for i in range(0, 10):
        transform = aug.get_random_transform((256, 256))
        print(transform)
        img = aug.apply_transform(img, transform)
        cv2.imwrite('{}.jpg'.format(i), img)


if __name__ == "__main__":
    main()
