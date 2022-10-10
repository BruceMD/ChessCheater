import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers, models
from keras.datasets import cifar10
from random import randint


def buildModel():

    training_images = np.load("training_images.npy")
    training_labels = np.load("training_labels.npy")
    testing_images = np.load("testing_images.npy")
    testing_labels = np.load("testing_labels.npy")

    print(training_images.shape)
    print(testing_images.shape)
    print()

    # class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    model = models.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(80, 80, 3)))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(96, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Flatten())         # converts it all into a 1D array (150, 150, 3) -> 67 500

    model.add(layers.Dense(96, activation="relu"))
    model.add(layers.Dense(12, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())

    model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print("Loss: {}".format(loss))
    print("Accuracy: {}".format(accuracy))

    model.save("chess_piece_classifier.model")


def trainingData():
    pass
# what's probably going to happen is that the training data will mostly be the same as the testing data
# the types of pieces are very finite as is
# get collection of all images, then scale them to all be the same size 80X80 pxels

