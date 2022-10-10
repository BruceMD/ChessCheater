import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers, models
from keras.datasets import cifar10
from random import randint


def buildModel():

    (training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()
    training_images, testing_images = training_images / 255, testing_images / 255

    class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    # for i in range(16):
    #     plt.subplot(4, 4, i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.imshow(training_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[training_labels[i][0]])
    # plt.show()

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(96, (3, 3), activation="relu"))
    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print("Loss: {}".format(loss))
    print("Accuracy: {}".format(accuracy))

    model.save("image_classifier.model")


def loadModel():
    model = models.load_model("image_classifier.model")

    (training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()
    training_images, testing_images = training_images / 255, testing_images / 255

    class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    for i in range(5):
        num = randint(0, testing_labels.size)
        print(num)

        prediction = model.predict(np.array([testing_images[num]]))
        index = np.argmax(prediction)

        # show the image
        plt.subplot(1, 2, 1)
        plt.imshow(testing_images[num])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(class_names[index])

        # show the graph scoring
        plt.subplot(1, 2, 2)
        plt.bar(class_names, prediction[0])
        plt.xlabel("Class names")
        plt.ylabel("Class scores")

        plt.show()
