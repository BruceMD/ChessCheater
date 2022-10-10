import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def labelDic(name):
    dic = {
        "pawn_w": 0,
        "pawn_b": 1,
        "knight_w": 2,
        "knight_b": 3,
        "bishop_w": 4,
        "bishop_b": 5,
        "rook_w": 6,
        "rook_b": 7,
        "queen_w": 8,
        "queen_b": 9,
        "king_w": 10,
        "king_b": 11
    }
    try:
        return dic[name]
    except KeyError:
        return None


def createTrainingDataset():
    path = "Training_Images"

    training_images, training_labels = [], []
    for img in os.listdir(path):
        pic = cv2.imread(os.path.join(path, img))
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic, (80, 80))
        training_images.append(pic)
        name = img[4:-5]
        training_labels.append(labelDic(name))

    np.save("training_images", np.array(training_images))
    np.save("training_labels", np.array(training_labels))


def createTestingDataset():
    path = "Testing_Images"

    testing_images, testing_labels = [], []
    for img in os.listdir(path):
        pic = cv2.imread(os.path.join(path, img))
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic, (80, 80))
        testing_images.append(pic)
        name = img[4:-5]
        testing_labels.append(labelDic(name))

    np.save("testing_images", np.array(testing_images))
    np.save("testing_labels", np.array(testing_labels))


def loadDataset():
    training_images = np.load("training_images.npy")
    training_images = training_images / 255
    training_labels = np.load("training_labels.npy")

    print(training_images.shape)
    print(training_labels.shape)

    for i in range(len(training_labels)):
        plt.subplot(6, 16, i + 1)
        plt.imshow(training_images[i].reshape(80, 80, 3))
        plt.xlabel(training_labels[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

    print(training_images)
    print(training_labels)
