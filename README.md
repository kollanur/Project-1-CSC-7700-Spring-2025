# Multilayer Perceptron (MLP) for MNIST Classification and Auto MPG Regression

## Overview

This repository contains the implementation of a Multilayer Perceptron (MLP) model for two different machine learning tasks:
- **Handwritten digit classification** using the MNIST dataset.
- **Vehicle fuel efficiency prediction** using the Auto MPG dataset.

The project includes model design, data preprocessing, training, evaluation, and visualization of results.

## Model Architecture

### MNIST Classification:
- Input Layer: **784 neurons** (flattened 28x28 image pixels)
- Hidden Layers:
  - **256 neurons** (ReLU)
  - **128 neurons** (ReLU)
  - **64 neurons** (ReLU)
  - **32 neurons** (ReLU)
- Output Layer: **10 neurons** (Softmax activation for classification)

### Auto MPG Regression:
- Input Layer: Features after preprocessing
- Hidden Layers:
  - **8 layers, each with 128 neurons** (ReLU)
- Output Layer: **1 neuron** (Linear activation for continuous regression)

## Installation & Requirements

## Usage

python train_mnist.py

python train_mpg.py


To run the project, ensure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn torch torchvision matplotlib prettytable```




