#!/bin/bash

bash utils/clean_project.sh

python src/dog_breeds/data_preprocessing.py
python src/dog_breeds/build_model.py
