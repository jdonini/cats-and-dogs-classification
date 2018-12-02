#!/bin/bash

bash utils/clean_project.sh

python src/species/data_preprocessing.py
python src/species/build_model.py
