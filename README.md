# Image Classification using CNN 🖼️🔍

This project implements an image classification system utilizing Convolutional Neural Networks (CNNs) in Python, leveraging TensorFlow and Keras libraries. The model is trained on a dataset containing 50,000 images from the CIFAR-10 dataset.

## Overview ℹ️

The aim of this project is to classify images into predefined classes using CNNs. CNNs are particularly effective in image classification tasks as they can automatically learn features from images. The project utilizes TensorFlow and Keras, which are popular deep learning frameworks, to build and train the CNN model.

## Dataset 📂

The dataset used for this project is the CIFAR-10 dataset, which consists of 50,000 32x32 color training images, labeled into 10 classes. The dataset can be downloaded from the following link:
[Download CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

## Model Architecture 🏗️

The CNN architecture employed in this project consists of multiple convolutional layers followed by max-pooling layers to extract features from the images. The final layers include fully connected layers with softmax activation for classification. 

## Results 📊

After training the model on the CIFAR-10 dataset, the model achieved a test dataset accuracy of 0.6973999738693237. This accuracy indicates the effectiveness of the model in accurately classifying images into their respective classes.

## Usage 🚀

To use this project, follow these steps:

1. Download the CIFAR-10 dataset from the provided link.
2. Preprocess the dataset as required.
3. Run the Python script to train the CNN model.
4. Evaluate the trained model using test data.

```bash
python train_model.py
```

## Dependencies 🛠️

- TensorFlow
- Keras
- NumPy
- Matplotlib

Install dependencies using pip:

```bash
pip install tensorflow keras numpy matplotlib
```
