# MNIST-classification-using-Tensorflow

-->Required Libraries

1.Numpy

2.Tensorflow

3.Pandas

--->MNIST dataset

The MNIST database is a large database of handwritten digits (image size = 28*28 ) that is commonly used for training various image processing systems. It contains 60,000 training images and 10,000 testing images. The dataset provided in this tutorial is .csv version of MNIST dataset. Each handwritten digit image is reshaped to size [1x784]. Hence the dataset provided has 785 columns where the first column is the label of image.

--->Working on project

The file NeuralNetwork.py is the main file. On running this file one will be offered with two choices, either to train the network or to use already saved trained model to make predictions. If the choice of prediction is made then the prediction will be saved in variable named ypred, which can be printed to display the prediction.

--->Results

This projects uses gradient descent algorithm and provides a training accuracy of 97.94%
