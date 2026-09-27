# Iris Flower Classification using Deep Learning

## Project Description

This project explores **multi-class classification of Iris flowers**
using a small feed-forward neural network built with TensorFlow/Keras.
It uses the classic Iris dataset and its four numeric
measurements---sepal length, sepal width, petal length, and petal
width---to predict the flower variety.

The notebook also trains a Scikit-learn **Perceptron** as a baseline,
then builds and evaluates a dense neural network. The workflow includes
exploratory data analysis, label encoding, a stratified train/test
split, feature scaling, one-hot encoding, model training, and
accuracy/loss visualization.

## Features

-   Explore the dataset with Pandas and a Seaborn pair plot.
-   Encode the categorical flower labels into numeric classes.
-   Split the data into training and testing sets while preserving class
    proportions.
-   Train a Scikit-learn Perceptron baseline.
-   Train a Keras neural network with ReLU hidden layers and a Softmax
    output layer.
-   Evaluate the model on the test set and plot training/validation
    accuracy.

## Model Architecture

The neural network in the notebook has: 1. Input: 4 features 2. Dense
hidden layer: 16 neurons, ReLU 3. Dense hidden layer: 8 neurons, ReLU 4.
Output layer: 3 neurons, Softmax

It is compiled with the Adam optimizer, categorical cross-entropy loss,
and accuracy as the metric. Training is configured for 100 epochs, batch
size 8, and a 20% validation split from the training data.

## Dataset

The notebook expects a file named `iris.csv` in the same working
directory as the notebook. It must contain four numeric feature columns
and a target column named `variety`, with the flower class labels.

## Requirements

-   Python 3
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   TensorFlow

Install the libraries with:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

## How to Run

1.  Clone or download this repository.

2.  Place `iris.csv` in the same directory as the notebook.

3.  Install the dependencies listed above.

4.  Launch Jupyter Notebook or JupyterLab:

    ``` bash
    jupyter notebook
    ```

5.  Open `Iris_prediction_dl_first.ipynb` and run the cells from top to
    bottom.

## Important Implementation Note

In the feature-scaling cell, the notebook currently calls
`fit_transform` on the test set. For a proper evaluation, fit the scaler
only on the training data and use `transform` on the test data:

``` python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This keeps the test set out of the scaler-fitting process and ensures
both sets use the same scaling parameters.

## Project Structure

``` text
.
├── Iris_prediction_dl_first.ipynb
├── iris.csv
└── README.md
```

## Learning Outcomes

-   Understand a basic supervised classification workflow.
-   Compare a linear Perceptron baseline with a multi-layer neural
    network.
-   Practice preprocessing and one-hot encoding for a multi-class
    target.
-   Train, validate, and evaluate a neural network using Keras.

## Future Improvements

-   Add a confusion matrix and per-class precision, recall, and F1-score
    for the neural network.
-   Add reproducibility controls for TensorFlow and compare results
    across multiple runs.
-   Save the trained model and provide a small prediction interface for
    new flower measurements.
