from random import shuffle
import cv2
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa


def build_example(label, image):
    feature = {
        'label': bytes_feature(tf.compat.as_bytes(label.tostring())),
        'image': bytes_feature(tf.compat.as_bytes(image.tostring()))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def build_sample(sign):
    image = cv2.imread(sign['file_path'])
    resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return resized


def crop_centered_image(image):
    (rows, cols, deeph) = image.shape
    if rows > cols:
        start = (rows - cols) // 2
        end = rows - start
        roi = image[start:end, :]
    else:
        start = (cols - rows) // 2
        end = cols - start
        roi = image[:, start:end]
    return roi


def write_dataset_file(filepath, samples, class_count):
    labels = np.eye(class_count, dtype=np.uint8)
    with tf.python_io.TFRecordWriter(filepath) as writer:
        for (category_int, filepath) in samples:
            image = cv2.imread(filepath)
            if image is None:
                print(filepath)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = crop_centered_image(image)
                image = cv2.resize(image, (128, 128))
                label = labels[category_int]
                # cv2.imshow('Image', image)
                # print(label)
                # cv2.waitKey(1)
                example = build_example(label, image)
                writer.write(example.SerializeToString())


def write_augmented_dataset_file(filepath, samples, class_count):
    augmentator = iaa.SomeOf((1, 4), [
        iaa.Fliplr(1.0),
        iaa.Add((-100, 100), per_channel=True),
        iaa.Sharpen(alpha=0.5),
        iaa.GaussianBlur(sigma=(0.0, 2.0)),
        iaa.Emboss(alpha=0.5),
        iaa.Invert(1.0, per_channel=1.0),
        iaa.ContrastNormalization((0.5, 2), per_channel=0.5),
        iaa.AdditiveGaussianNoise(scale=0.1 * 255, per_channel=0.5),
        iaa.CoarseDropout((0.1, 0.3), size_percent=(0.05, 0.2), per_channel=0.5),
        iaa.Crop(percent=(0, 0.07)),
        iaa.Dropout((0.1, 0.3), per_channel=0.5),
        iaa.Grayscale(alpha=(0.2, 1.0)),
        iaa.ElasticTransformation(alpha=(0.0, 3.0), sigma=0.25),
        iaa.Affine(rotate=(-30, 30)),
        iaa.Affine(shear=(-30, 30)),
    ], random_order=True)
    labels = np.eye(class_count, dtype=np.uint8)
    with tf.python_io.TFRecordWriter(filepath) as writer:
        for (category_int, filepath) in samples:
            image = cv2.imread(filepath)
            if image is None:
                print(filepath)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = crop_centered_image(image)
                image = cv2.resize(image, (128, 128))
                [image] = augmentator.augment_images([image])
                label = labels[category_int]
                # cv2.imshow('Image', image)
                # print(label)
                # cv2.waitKey(1)
                example = build_example(label, image)
                writer.write(example.SerializeToString())


def main():
    category_to_int = {
        'abyssinian': 1,
        'american_bulldog': 2,
        'american_pit_bull_terrier': 3,
        'basset_hound': 4,
        'beagle': 5,
        'bengal': 6,
        'birman': 7,
        'bombay': 8,
        'boxer': 9,
        'british_shorthair': 10,
        'chihuahua': 11,
        'egyptian_mau': 12,
        'english_cocker_spaniel': 13,
        'english_setter': 14,
        'german_shorthaired': 15,
        'great_pyrenees': 16,
        'havanese': 17,
        'japanese_chin': 18,
        'keeshond': 19,
        'leonberger': 20,
        'maine_coon': 21,
        'miniature_pinscher': 22,
        'newfoundland': 23,
        'persian': 24,
        'pomeranian': 25,
        'pug': 26,
        'ragdoll': 27,
        'russian_blue': 28,
        'saint_bernard': 29,
        'samoyed': 30,
        'scottish_terrier': 31,
        'shiba_inu': 32,
        'siamese': 33,
        'sphynx': 34,
        'staffordshire_bull_terrier': 35,
        'wheaten_terrier': 36,
        'yorkshire_terrier': 37
    }

    int_to_category = {i: category for (category, i) in category_to_int.items()}
    categories = {category: [] for category in category_to_int.keys()}

    with open('../datasets/dataset/annotations/list.txt', 'rt') as lines:
        for line in lines:
            if line[0] == '#':
                pass
            else:
                (file_path, class_id, category, *tail) = line.split(' ')
                complete_file_path = './../datasets/dataset/images/{}.jpg'.format(file_path)
                categories[int_to_category[int(class_id)]].append(complete_file_path)

    samples_count = min([len(file_paths) for file_paths in categories.values()])
    train_count = int(samples_count * 0.7)
    test_count = int(samples_count * 0.3)

    for samples in categories.values():
        shuffle(samples)
    augmentation_factor = 16
    train_samples = [
            (category_to_int[category] - 1, sample)
            for (category, samples) in categories.items()
            for sample in samples[:train_count]
            for i in range(augmentation_factor)]
    test_samples = [
            (category_to_int[category] - 1, sample)
            for (category, samples) in categories.items()
            for sample in samples[train_count:]]
    shuffle(train_samples)
    shuffle(test_samples)
    print('Training')
    write_dataset_file('./dataset/train.tfrecords', train_samples, len(categories))
    print('Testing')
    write_dataset_file('./dataset/test.tfrecords', test_samples, len(categories))


if __name__ == '__main__':
    main()
