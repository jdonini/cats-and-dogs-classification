#!/bin/bash

bash utils/clean_project.sh

python src/cat_breeds/data_preprocessing.py
python src/cat_breeds/build_model.py
