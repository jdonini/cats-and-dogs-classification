#!/bin/bash

mkdir database

cd database
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar xzvf images.tar.gz
rm images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar xzvf annotations.tar.gz
rm annotations.tar.gz


# mkdir models
# mkdir models/breeds
# mkdir models/species
# mkdir models/dogs
# mkdir models/cats
# mkdir database
# mkdir database/dataset

# mkdir database/data_species
# mkdir database/data_species/test
# mkdir database/data_species/train
# mkdir database/data_species/train/cat
# mkdir database/data_species/train/dog
# mkdir database/data_species/test/cat
# mkdir database/data_species/test/dog

# mkdir database/data_dogs
# mkdir database/data_dogs/train
# mkdir database/data_dogs/test
# mkdir database/data_cats

# mkdir database/data_cats/train
# mkdir database/data_cats/test

# mkdir database/data_cats/train/abyssinian
# mkdir database/data_cats/train/bengal
# mkdir database/data_cats/train/birman
# mkdir database/data_cats/train/bombay
# mkdir database/data_cats/train/british_shorthair
# mkdir database/data_cats/train/egyptian_mau
# mkdir database/data_cats/train/maine_coon
# mkdir database/data_cats/train/persian
# mkdir database/data_cats/train/ragdoll
# mkdir database/data_cats/train/russian_blue
# mkdir database/data_cats/train/siamese
# mkdir database/data_cats/train/sphynx

# mkdir database/data_cats/test/abyssinian
# mkdir database/data_cats/test/bengal
# mkdir database/data_cats/test/birman
# mkdir database/data_cats/test/bombay
# mkdir database/data_cats/test/british_shorthair
# mkdir database/data_cats/test/egyptian_mau
# mkdir database/data_cats/test/maine_coon
# mkdir database/data_cats/test/persian
# mkdir database/data_cats/test/ragdoll
# mkdir database/data_cats/test/russian_blue
# mkdir database/data_cats/test/siamese
# mkdir database/data_cats/test/sphynx

# mkdir database/data_dogs/train/american_bulldog
# mkdir database/data_dogs/train/american_pit_bull_terrier
# mkdir database/data_dogs/train/basset_hound
# mkdir database/data_dogs/train/beagle
# mkdir database/data_dogs/train/boxer
# mkdir database/data_dogs/train/chihuahua
# mkdir database/data_dogs/train/english_cocker_spaniel
# mkdir database/data_dogs/train/english_setter
# mkdir database/data_dogs/train/german_shorthaired
# mkdir database/data_dogs/train/great_pyrenees
# mkdir database/data_dogs/train/havanese
# mkdir database/data_dogs/train/japanese_chin
# mkdir database/data_dogs/train/keeshond
# mkdir database/data_dogs/train/leonberger
# mkdir database/data_dogs/train/miniature_pinscher
# mkdir database/data_dogs/train/newfoundland
# mkdir database/data_dogs/train/pomeranian
# mkdir database/data_dogs/train/pug
# mkdir database/data_dogs/train/saint_bernard
# mkdir database/data_dogs/train/samoyed
# mkdir database/data_dogs/train/scottish_terrier
# mkdir database/data_dogs/train/shiba_inu
# mkdir database/data_dogs/train/staffordshire_bull_terrier
# mkdir database/data_dogs/train/wheaten_terrier
# mkdir database/data_dogs/train/yorkshire_terrier

# mkdir database/data_dogs/test/american_bulldog
# mkdir database/data_dogs/test/american_pit_bull_terrier
# mkdir database/data_dogs/test/basset_hound
# mkdir database/data_dogs/test/beagle
# mkdir database/data_dogs/test/boxer
# mkdir database/data_dogs/test/chihuahua
# mkdir database/data_dogs/test/english_cocker_spaniel
# mkdir database/data_dogs/test/english_setter
# mkdir database/data_dogs/test/german_shorthaired
# mkdir database/data_dogs/test/great_pyrenees
# mkdir database/data_dogs/test/havanese
# mkdir database/data_dogs/test/japanese_chin
# mkdir database/data_dogs/test/keeshond
# mkdir database/data_dogs/test/leonberger
# mkdir database/data_dogs/test/miniature_pinscher
# mkdir database/data_dogs/test/newfoundland
# mkdir database/data_dogs/test/pomeranian
# mkdir database/data_dogs/test/pug
# mkdir database/data_dogs/test/saint_bernard
# mkdir database/data_dogs/test/samoyed
# mkdir database/data_dogs/test/scottish_terrier
# mkdir database/data_dogs/test/shiba_inu
# mkdir database/data_dogs/test/staffordshire_bull_terrier
# mkdir database/data_dogs/test/wheaten_terrier
# mkdir database/data_dogs/test/yorkshire_terrier

# mkdir database/data_breeds
# mkdir database/data_breeds/train
# mkdir database/data_breeds/train/american_bulldog
# mkdir database/data_breeds/train/american_pit_bull_terrier
# mkdir database/data_breeds/train/basset_hound
# mkdir database/data_breeds/train/beagle
# mkdir database/data_breeds/train/boxer
# mkdir database/data_breeds/train/chihuahua
# mkdir database/data_breeds/train/english_cocker_spaniel
# mkdir database/data_breeds/train/english_setter
# mkdir database/data_breeds/train/german_shorthaired
# mkdir database/data_breeds/train/great_pyrenees
# mkdir database/data_breeds/train/havanese
# mkdir database/data_breeds/train/japanese_chin
# mkdir database/data_breeds/train/keeshond
# mkdir database/data_breeds/train/leonberger
# mkdir database/data_breeds/train/miniature_pinscher
# mkdir database/data_breeds/train/newfoundland
# mkdir database/data_breeds/train/pomeranian
# mkdir database/data_breeds/train/pug
# mkdir database/data_breeds/train/saint_bernard
# mkdir database/data_breeds/train/samoyed
# mkdir database/data_breeds/train/scottish_terrier
# mkdir database/data_breeds/train/shiba_inu
# mkdir database/data_breeds/train/staffordshire_bull_terrier
# mkdir database/data_breeds/train/wheaten_terrier
# mkdir database/data_breeds/train/yorkshire_terrier
# mkdir database/data_breeds/train/abyssinian
# mkdir database/data_breeds/train/bengal
# mkdir database/data_breeds/train/birman
# mkdir database/data_breeds/train/bombay
# mkdir database/data_breeds/train/british_shorthair
# mkdir database/data_breeds/train/egyptian_mau
# mkdir database/data_breeds/train/maine_coon
# mkdir database/data_breeds/train/persian
# mkdir database/data_breeds/train/ragdoll
# mkdir database/data_breeds/train/russian_blue
# mkdir database/data_breeds/train/siamese
# mkdir database/data_breeds/train/sphynx

# mkdir database/data_breeds/test
# mkdir database/data_breeds/test/american_bulldog
# mkdir database/data_breeds/test/american_pit_bull_terrier
# mkdir database/data_breeds/test/basset_hound
# mkdir database/data_breeds/test/beagle
# mkdir database/data_breeds/test/boxer
# mkdir database/data_breeds/test/chihuahua
# mkdir database/data_breeds/test/english_cocker_spaniel
# mkdir database/data_breeds/test/english_setter
# mkdir database/data_breeds/test/german_shorthaired
# mkdir database/data_breeds/test/great_pyrenees
# mkdir database/data_breeds/test/havanese
# mkdir database/data_breeds/test/japanese_chin
# mkdir database/data_breeds/test/keeshond
# mkdir database/data_breeds/test/leonberger
# mkdir database/data_breeds/test/miniature_pinscher
# mkdir database/data_breeds/test/newfoundland
# mkdir database/data_breeds/test/pomeranian
# mkdir database/data_breeds/test/pug
# mkdir database/data_breeds/test/saint_bernard
# mkdir database/data_breeds/test/samoyed
# mkdir database/data_breeds/test/scottish_terrier
# mkdir database/data_breeds/test/shiba_inu
# mkdir database/data_breeds/test/staffordshire_bull_terrier
# mkdir database/data_breeds/test/wheaten_terrier
# mkdir database/data_breeds/test/yorkshire_terrier
# mkdir database/data_breeds/test/abyssinian
# mkdir database/data_breeds/test/bengal
# mkdir database/data_breeds/test/birman
# mkdir database/data_breeds/test/bombay
# mkdir database/data_breeds/test/british_shorthair
# mkdir database/data_breeds/test/egyptian_mau
# mkdir database/data_breeds/test/maine_coon
# mkdir database/data_breeds/test/persian
# mkdir database/data_breeds/test/ragdoll
# mkdir database/data_breeds/test/russian_blue
# mkdir database/
# mkdir database/data_breeds/
# mkdir database/data_cats/
# mkdir database/data_dogs/
# mkdir database/data_species/
# mkdir database/dataset/

cd database/dataset
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar xzvf images.tar.gz
rm images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar xzvf annotations.tar.gz
rm annotations.tar.gz
