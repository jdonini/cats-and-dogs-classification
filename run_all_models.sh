#!/bin/bash

bash utils/clean_project.sh

python src/breeds/build_model.py
python src/cat_breeds/build_model.py
python src/dog_breeds/build_model.py
python src/species/build_model.py
