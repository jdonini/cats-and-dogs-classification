import os

from scipy.misc import imsave
from imgaug import augmenters as iaa
from config.config import DATA_PATH_IMAGES


def augmentator(images):
    return iaa.SomeOf(
        (1, 2),
        [
            iaa.Fliplr(1.0),
            iaa.Add((-50, 50), per_channel=True),
            iaa.Sharpen(alpha=0.5),
            iaa.GaussianBlur(sigma=(0.0, 2.0)),
            iaa.Emboss(alpha=0.5),
            iaa.ContrastNormalization((0.5, 2), per_channel=0.5),
            iaa.AdditiveGaussianNoise(scale=0.1 * 255, per_channel=0.5),
            iaa.CoarseDropout(0.01, size_percent=0.1, per_channel=1),
            iaa.Crop(percent=(0, 0.07)),
            iaa.ContrastNormalization((0.75, 1.5)),
            iaa.Dropout((0.02, 0.1), per_channel=0.5),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.ElasticTransformation(alpha=(0.0, 3.0), sigma=0.25),
            iaa.Sequential(
                [
                    iaa.Affine(translate_px={"x": -20}),
                    iaa.AdditiveGaussianNoise(scale=0.1 * 255),
                ]
            ),
            iaa.Affine(shear=(-16, 16)),
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},),
            iaa.Affine(rotate=(-16, 16),),
            iaa.Affine(scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},),
        ],
        random_order=True,
    )


def augment_images(image, f):
    images = [image] * 16
    images_augmented = augmentator(images)
    for i, images_augmented in enumerate(images_augmented):
        imsave(f + "_%06d.jpg" % (i,), images_augmented)


def remove_noise():
    print("\nProcessing remove noise \n")
    for root, _, files in os.walk(DATA_PATH_IMAGES):
        for currentFile in files:
            print("processing file: " + currentFile)
            exts = ".mat"
            if any(currentFile.lower().endswith(ext) for ext in exts):
                os.remove(os.path.join(root, currentFile))
