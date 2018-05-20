#!/bin/bash

python scripts/breeds/build_model.py && \
python scripts/cat_breeds/build_model.py && \
python scripts/dog_breeds/build_model.py && \
python scripts/species/build_model.py
