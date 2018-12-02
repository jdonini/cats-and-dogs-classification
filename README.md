# Cats and Dogs Classification

The Oxford-IIIT Pet Dataset.
The problem is to classify each breed of animal presented in the dataset.
The first step was to classify breeds between dogs and cats, after doing this the breeds of dogs and cats were classified separatelythe, and finally, mixed the races and made the classification, increasing the degree of difficulty of problem.

## Step 1

get dataset:

- bash utils/get_dataset.sh

## Step 2

preprocessing the dataset:

- bash rul_all_preprocessing.sh

## Step 3

Creation of the training model:

- bash run_all_models.sh

## Step 4

To run the TensorBoard, open a new terminal and run the command below. Then, open http://localhost:6006/ in your web browser.

- script/
- choose your model
- tensorboard --logdir='./logs' --port=6006
