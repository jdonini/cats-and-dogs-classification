import random
import time
import sys
import os
from PIL import Image
import numpy as np
from scipy.misc import imsave, imread
sys.path.append('utils')
from config import *
from data_augmentation import *

print("\nPreprocessing Breeds...")

train_samples, test_samples = [], []

breeds = {
    'abyssinian': [],
    'american_bulldog': [],
    'american_pit_bull_terrier': [],
    'basset_hound': [],
    'beagle': [],
    'bengal': [],
    'birman': [],
    'bombay': [],
    'boxer': [],
    'british_shorthair': [],
    'chihuahua': [],
    'egyptian_mau': [],
    'english_cocker_spaniel': [],
    'english_setter': [],
    'german_shorthaired': [],
    'great_pyrenees': [],
    'havanese': [],
    'japanese_chin': [],
    'keeshond': [],
    'leonberger': [],
    'maine_coon': [],
    'miniature_pinscher': [],
    'newfoundland': [],
    'persian': [],
    'pomeranian': [],
    'pug': [],
    'ragdoll': [],
    'russian_blue': [],
    'saint_bernard': [],
    'samoyed': [],
    'scottish_terrier': [],
    'shiba_inu': [],
    'siamese': [],
    'sphynx': [],
    'staffordshire_bull_terrier': [],
    'wheaten_terrier': [],
    'yorkshire_terrier': []
}

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

int_to_category = {
      1: 'abyssinian',
      2: 'american_bulldog',
      3: 'american_pit_bull_terrier',
      4:  'basset_hound',
      5:  'beagle',
      6:  'bengal',
      7:  'birman',
      8:  'bombay',
      9:  'boxer',
      10: 'british_shorthair',
      11: 'chihuahua',
      12: 'egyptian_mau',
      13: 'english_cocker_spaniel',
      14: 'english_setter',
      15: 'german_shorthaired',
      16: 'great_pyrenees',
      17: 'havanese',
      18: 'japanese_chin',
      19: 'keeshond',
      20: 'leonberger',
      21: 'maine_coon',
      22: 'miniature_pinscher',
      23: 'newfoundland',
      24: 'persian',
      25: 'pomeranian',
      26: 'pug',
      27: 'ragdoll',
      28: 'russian_blue',
      29: 'saint_bernard',
      30: 'samoyed',
      31: 'scottish_terrier',
      32: 'shiba_inu',
      33: 'siamese',
      34: 'sphynx',
      35: 'staffordshire_bull_terrier',
      36: 'wheaten_terrier',
      37: 'yorkshire_terrier'
}

dog_breeds = {
      1: 'american_bulldog',
      2: 'american_pit_bull_terrier',
      3: 'basset_hound',
      4: 'beagle',
      5: 'boxer',
      6: 'chihuahua',
      7: 'english_cocker_spaniel',
      8: 'english_setter',
      9: 'german_shorthaired',
      10: 'great_pyrenees',
      11: 'havanese',
      12: 'japanese_chin',
      13: 'keeshond',
      14: 'leonberger',
      15: 'miniature_pinscher',
      16: 'newfoundland',
      17: 'pomeranian',
      18: 'pug',
      19: 'saint_bernard',
      20: 'samoyed',
      21: 'scottish_terrier',
      22: 'shiba_inu',
      23: 'staffordshire_bull_terrier',
      24: 'wheaten_terrier',
      25: 'yorkshire_terrier'
}

cat_breeds = {
      1: 'abyssinian',
      2: 'bengal',
      3: 'birman',
      4: 'bombay',
      5: 'british_shorthair',
      6: 'egyptian_mau',
      7: 'maine_coon',
      8: 'persian',
      9: 'ragdoll',
      10: 'russian_blue',
      11: 'siamese',
      12: 'sphynx'
}

with open(DATA_ANNOTATION, 'rt') as lines:
    for line in lines:
        if line[0] == '#':
            pass
        else:
            (file_path, class_id, category, *tail) = line.split(' ')
            complete_file_path = DATA_PATH_IMAGES+'{}.jpg'.format(file_path)
            breeds[int_to_category[int(class_id)]].append(file_path)

samples_count = min([len(file_paths) for file_paths in breeds.values()])
train_count = int(samples_count * 0.7)
test_count = int(samples_count * 0.3)

for (class_id, file_paths) in breeds.items():
    random.shuffle(file_paths)
    for file_path in file_paths[:train_count]:
        train_samples.append((class_id, file_path))
    for file_path in file_paths[train_count:train_count + test_count]:
        test_samples.append((class_id, file_path))

random.shuffle(train_samples)
random.shuffle(test_samples)


def all_data_augmentation():
    augment_images(image, f)


remove_noise()

print('\nProcessing train samples...')
time_start_train = time.time()
for (class_id, file_path) in train_samples:
    for item in dirs:
        if (item.split('.')[0] == file_path) and (class_id == dog_breeds[1]):
            f, e = os.path.splitext(SAVE_AMERICA_BULLDOG_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[2]):
            f, e = os.path.splitext(SAVE_AMERICAN_PIT_TUL_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[3]):
            f, e = os.path.splitext(SAVE_BASSET_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[4]):
            f, e = os.path.splitext(SAVE_BEAGLE_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[5]):
            f, e = os.path.splitext(SAVE_BOXER_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[6]):
            f, e = os.path.splitext(SAVE_CHIHUAHUA_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[7]):
            f, e = os.path.splitext(SAVE_ENGLISH_COCKER_SPANIEL_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[8]):
            f, e = os.path.splitext(SAVE_ENGLISH_SETTER_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[9]):
            f, e = os.path.splitext(SAVE_GERMAN_SHORTRED_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[10]):
            f, e = os.path.splitext(SAVE_GREAT_PYRENESS_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[11]):
            f, e = os.path.splitext(SAVE_HAVANESE_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[12]):
            f, e = os.path.splitext(SAVE_JAPANESE_CHIN_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[13]):
            f, e = os.path.splitext(SAVE_KEESHOND_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[14]):
            f, e = os.path.splitext(SAVE_LEONBERGER_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[15]):
            f, e = os.path.splitext(SAVE_MINIATURE_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[16]):
            f, e = os.path.splitext(SAVE_NEWFOUNDLAND_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[17]):
            f, e = os.path.splitext(SAVE_POMERANIAN_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[18]):
            f, e = os.path.splitext(SAVE_PUG_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[19]):
            f, e = os.path.splitext(SAVE_SAINT_BERNARD_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[20]):
            f, e = os.path.splitext(SAVE_SAMOYED_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[21]):
            f, e = os.path.splitext(SAVE_SCOTTISH_TERRIER_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[22]):
            f, e = os.path.splitext(SAVE_SHIBA_INU_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[23]):
            f, e = os.path.splitext(SAVE_STAFFORDSHIRE_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[24]):
            f, e = os.path.splitext(SAVE_WHEATEN_TERRIER_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[25]):
            f, e = os.path.splitext(SAVE_YORKSHIRE_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[1]):
            f, e = os.path.splitext(SAVE_ABYSSIANIAN_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[2]):
            f, e = os.path.splitext(SAVE_BENGAL_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[3]):
            f, e = os.path.splitext(SAVE_BIRMAN_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[4]):
            f, e = os.path.splitext(SAVE_BOMBAY_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[5]):
            f, e = os.path.splitext(SAVE_BRITISH_SHORTHAIR_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[6]):
            f, e = os.path.splitext(SAVE_EGYPTIAN_MAU_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[7]):
            f, e = os.path.splitext(SAVE_MAINE_COON_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[8]):
            f, e = os.path.splitext(SAVE_PERSIAN_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[9]):
            f, e = os.path.splitext(SAVE_RAGDOOL_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[10]):
            f, e = os.path.splitext(SAVE_RUSSIAN_BLUE_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[11]):
            f, e = os.path.splitext(SAVE_SIAMESE_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[12]):
            f, e = os.path.splitext(SAVE_SPHYNX_TRAIN + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
            all_data_augmentation()
time_test = time.time() - time_start_train
print('Time to process train samples: {:.2f} [sec].'.format(time_test))

print('\nProcessing test samples...')
time_start_test = time.time()
for (class_id, file_path) in test_samples:
    for item in dirs:
        if (item.split('.')[0] == file_path) and (class_id == dog_breeds[1]):
            f, e = os.path.splitext(SAVE_AMERICA_BULLDOG_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[2]):
            f, e = os.path.splitext(SAVE_AMERICAN_PIT_TUL_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[3]):
            f, e = os.path.splitext(SAVE_BASSET_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[4]):
            f, e = os.path.splitext(SAVE_BEAGLE_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[5]):
            f, e = os.path.splitext(SAVE_BOXER_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[6]):
            f, e = os.path.splitext(SAVE_CHIHUAHUA_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[7]):
            f, e = os.path.splitext(SAVE_ENGLISH_COCKER_SPANIEL_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[8]):
            f, e = os.path.splitext(SAVE_ENGLISH_SETTER_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[9]):
            f, e = os.path.splitext(SAVE_GERMAN_SHORTRED_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[10]):
            f, e = os.path.splitext(SAVE_GREAT_PYRENESS_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[11]):
            f, e = os.path.splitext(SAVE_HAVANESE_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[12]):
            f, e = os.path.splitext(SAVE_JAPANESE_CHIN_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[13]):
            f, e = os.path.splitext(SAVE_KEESHOND_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[14]):
            f, e = os.path.splitext(SAVE_LEONBERGER_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[15]):
            f, e = os.path.splitext(SAVE_MINIATURE_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[16]):
            f, e = os.path.splitext(SAVE_NEWFOUNDLAND_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[17]):
            f, e = os.path.splitext(SAVE_POMERANIAN_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[18]):
            f, e = os.path.splitext(SAVE_PUG_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[19]):
            f, e = os.path.splitext(SAVE_SAINT_BERNARD_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[20]):
            f, e = os.path.splitext(SAVE_SAMOYED_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[21]):
            f, e = os.path.splitext(SAVE_SCOTTISH_TERRIER_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[22]):
            f, e = os.path.splitext(SAVE_SHIBA_INU_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[23]):
            f, e = os.path.splitext(SAVE_STAFFORDSHIRE_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[24]):
            f, e = os.path.splitext(SAVE_WHEATEN_TERRIER_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id == dog_breeds[25]):
            f, e = os.path.splitext(SAVE_YORKSHIRE_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[1]):
            f, e = os.path.splitext(SAVE_ABYSSIANIAN_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[2]):
            f, e = os.path.splitext(SAVE_BENGAL_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[3]):
            f, e = os.path.splitext(SAVE_BIRMAN_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[4]):
            f, e = os.path.splitext(SAVE_BOMBAY_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[5]):
            f, e = os.path.splitext(SAVE_BRITISH_SHORTHAIR_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[6]):
            f, e = os.path.splitext(SAVE_EGYPTIAN_MAU_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[7]):
            f, e = os.path.splitext(SAVE_MAINE_COON_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[8]):
            f, e = os.path.splitext(SAVE_PERSIAN_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[9]):
            f, e = os.path.splitext(SAVE_RAGDOOL_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[10]):
            f, e = os.path.splitext(SAVE_RUSSIAN_BLUE_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[11]):
            f, e = os.path.splitext(SAVE_SIAMESE_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
        elif (item.split('.')[0] == file_path) and (class_id in cat_breeds[12]):
            f, e = os.path.splitext(SAVE_SPHYNX_TEST + item)
            img = Image.open(DATA_PATH_IMAGES + item).convert("RGB")
            image = np.array(img)
            imsave(f + '.jpg', image)
time_train = time.time() - time_start_test
print('Time to process test samples: {:.2f} [sec].'.format(time_train))

print('\nTime to process all stages: {:.2f} [sec].'.format(time_test + time_train))
