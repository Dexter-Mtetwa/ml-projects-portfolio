# Digit Recognizer with Deep Learning

A deep learning project that recognizes handwritten digits using a neural network built with TensorFlow/Keras.

## Overview

This project trains a neural network to classify handwritten digits from image data. The workflow covers data preprocessing, neural network construction, training, validation, and performance evaluation.

## Dataset

The project uses a digit recognition dataset containing images of handwritten numbers.

Each image:

* 28 × 28 pixels
* Flattened into 784 input features
* Target labels from 0–9

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* TensorFlow
* Keras

## Workflow

### Data Preparation

* Load image dataset
* Separate labels from pixel values
* Normalize pixel intensities
* Create training and validation sets

### Model Architecture

Neural network built using:

* Dense layers
* ReLU activations
* Output layer with 10 classes

### Training

* Forward propagation
* Backpropagation
* Validation monitoring

### Evaluation

Metrics include:

* Accuracy
* Validation performance
* Prediction visualization

## Project Structure

```text
digitRecog-dl.ipynb
README.md
```

## Key Learning Outcomes

* Neural network fundamentals
* Image classification
* TensorFlow/Keras workflow
* Data normalization techniques

## Future Improvements

* Convolutional Neural Networks (CNNs)
* Data augmentation
* Hyperparameter tuning
* Model deployment

## Author

Dexter
