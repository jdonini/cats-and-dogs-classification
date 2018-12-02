#!/bin/bash

bash utils/clean_project.sh

python src/breeds/data_preprocessing.py
python src/breeds/build_model.py
