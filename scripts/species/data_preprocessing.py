import random
import time
import sys
import os
from PIL import Image
import numpy as np
from scipy.misc import imsave, imread
sys.path.append('../../utils')
from config import *
from data_augmentation import *

print("\nPreprocessing Species...")

train_samples, test_samples = [], []

categories = {
    'cat': [],
    'dog': []
}

category_to_int = {
    'cat': 1,
    'dog': 2
}

int_to_category = {
    1: 'cat',
    2: 'dog'
}

with open(DATA_ANNOTATION, 'rt') as lines:
    for line in lines:
        if line[0] == '#':
            pass
        else:
            (file_path, class_id, category, *tail) = line.split(' ')
            complete_file_path = DATA_PATH_IMAGES+'{}.jpg'.format(file_path)
            categories[int_to_category[int(category)]].append(file_path)

samples_count = min([len(file_paths) for file_paths in categories.values()])
train_count = int(samples_count * 0.7)
test_count = int(samples_count * 0.3)

for (category, file_paths) in categories.items():
    random.shuffle(file_paths)
    for file_path in file_paths[:train_count]:
        train_samples.append((category, file_path))
    for file_path in file_paths[train_count:train_count + test_count]:
        test_samples.append((category, file_path))

random.shuffle(train_samples)
random.shuffle(test_samples)


def all_data_augmentation():
    augment_images(image, f)


remove_noise()

print('\nProcessing train samples...')
time_start_train = time.time()
for (category, file_path) in train_samples:
    for item in dirs:
            if item.split('.')[0] == file_path and category == 'dog':
                f, e = os.path.splitext(SAVE_DATASET_TRAIN_DOG + item)
                img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
                image = np.array(img)
                imsave(f + '.jpg', image)
                all_data_augmentation()
            elif item.split('.')[0] == file_path and category == 'cat':
                f, e = os.path.splitext(SAVE_DATASET_TRAIN_CAT + item)
                img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
                image = np.array(img)
                imsave(f + '.jpg', image)
                all_data_augmentation()
time_test = time.time() - time_start_train
print('Time to process train samples: {:.2f} [sec].'.format(time_test))

print('\nProcessing test samples...')
time_start_test = time.time()
for (category, file_path) in test_samples:
    for item in dirs:
            if item.split('.')[0] == file_path and category == 'dog':
                f, e = os.path.splitext(SAVE_DATASET_TEST_DOG + item)
                img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
                image = np.array(img)
                imsave(f + '.jpg', image)
            elif item.split('.')[0] == file_path and category == 'cat':
                f, e = os.path.splitext(SAVE_DATASET_TEST_CAT + item)
                img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
                image = np.array(img)
                imsave(f + '.jpg', image)
time_train = time.time() - time_start_test
print('Time to process test samples: {:.2f} [sec].'.format(time_train))

print('\nTime to process all stages: {:.2f} [sec].'.format(time_test + time_train))
