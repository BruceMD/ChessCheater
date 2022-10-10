import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def createDataset():
    path = "Training_Images/all"

    training_images, training_labels = [], []
    for img in os.listdir(path):
        pic = cv2.imread(os.path.join(path, img))
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic, (80, 80))
        training_images.append([pic])
        name = img[4:-5]
        training_labels.append(name)

    np.save("training_images", np.array(training_images))
    np.save("training_labels", np.array(training_labels))


def loadDataset():
    training_images = np.load("training_images.npy")
    training_labels = np.load("training_labels.npy")

    for i in range(len(training_labels)):
        plt.imshow(training_images[i].reshape(80, 80, 3))
        plt.xlabel(training_labels[i])
        plt.xticks([])
        plt.yticks([])
        plt.show()
