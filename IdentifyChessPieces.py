import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers, models
from keras.datasets import cifar10
from random import randint
from BuildImageDataset import labelDic, revLabelDic
from keras.preprocessing.image import ImageDataGenerator
import scipy


def buildModel():

    training_images = np.load("training_images.npy")
    training_labels = np.load("training_labels.npy")
    testing_images = np.load("testing_images.npy")
    testing_labels = np.load("testing_labels.npy")

    training_images = training_images / 255
    testing_images = testing_images / 255

    model = models.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(80, 80, 3)))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(96, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Flatten())         # converts it all into a 1D array (150, 150, 3) -> 67 500

    model.add(layers.Dense(104, activation="relu"))          # potentially make this 104 = 13 * 8
    model.add(layers.Dense(13, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())

    model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
    # model.fit(datagen.flow(training_images, training_labels, batch_size=batchSize), epochs=10,
    #           steps_per_epoch=training_images.shape[0] // batchSize, validation_data=(testing_images, testing_labels))

    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print("Loss: {}".format(loss))
    print("Accuracy: {}".format(accuracy))

    model.save("chess_piece_classifier.model")


def displayModel():
    identify = models.load_model("chess_piece_classifier.model")

    testing_images = np.load("testing_images.npy")
    print(testing_images.shape)

    for num in range(testing_images.shape[0]):
        prediction = identify.predict(np.array([testing_images[num]]))
        index = np.argmax(prediction)

        # show the image
        plt.subplot(7, 4, num+1)
        plt.imshow(testing_images[num])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(revLabelDic(index))

    plt.show()
