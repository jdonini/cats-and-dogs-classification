# Cats and Dogs Classification
The Oxford-IIIT Pet Dataset.
The problem is to classify each breed of animal presented in the dataset.
The first step was to classify breeds between dogs and cats, after doing this the breeds of dogs and cats were classified separatelythe, and finally, mixed the races and made the classification, increasing the degree of difficulty of problem.


## Step 1
Run get_dataset
- utils/
- bash get_dataset.sh

## Step 2
Run the preprocessing of the desired class
- script/
- python data_preprocessing.py

## Step 3
Run the model of the desired class
- script/
- python build_model.py

## Step 4
To run the TensorBoard, open a new terminal and run the command below. Then, open http://localhost:6006/ in your web browser.
- script/
- choose your model
- tensorboard --logdir='./logs' --port=6006
