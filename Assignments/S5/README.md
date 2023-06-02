# MNIST PyTorch Model
This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for image classification on the MNIST dataset.

## Overview
The MNIST dataset is a widely-used benchmark dataset in the field of machine learning. It consists of a collection of 60,000 handwritten digits from 0 to 9, each represented as a 28x28 grayscale image. The goal of this project is to train a CNN model that can accurately classify these digits.

## Files
This repository contains the following files:

### S5.ipynb: 
This Jupyter Notebook file contains the code for training and evaluating the MNIST PyTorch model. It provides a step-by-step guide for loading the dataset, defining the model architecture, training the model, and evaluating its performance. The notebook also includes sections for visualizing the training process.

### model.py: 
This Python file defines the architecture of the CNN model. It uses the PyTorch library to create a custom neural network with convolutional, pooling, and fully connected layers.

### utils.py: 
This Python file contains utility functions used in the training and evaluation process. It includes functions for training, testing and plotting the loss and accurcay.

## Usage
To use the MNIST PyTorch model, follow these steps:

* Clone this repository to your local machine.
* Open the S5.ipynb notebook using Colab Notebook or any compatible environment.
* Run the notebook cells sequentially to load the dataset, define the model, train the model, and evaluate its performance.

## Acknowledgments
The MNIST dataset was originally created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. More information about the dataset can be found at http://yann.lecun.com/exdb/mnist/. The dataset is made available under a Creative Commons Attribution-Share Alike 3.0 license.
