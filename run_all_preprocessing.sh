#!/bin/bash

python scripts/breeds/data_preprocessing.py && \
python scripts/cat_breeds/data_preprocessing.py && \
python scripts/dog_breeds/data_preprocessing.py && \
python scripts/species/data_preprocessing.py
