import os

import numpy as np
import pyautogui as pag
from time import sleep
import matplotlib.pyplot as plt


def findGrid():
    screenImage = np.array(pag.screenshot())

    cellColour = [49, 46, 43]                                           # assume this grey
    maxSide = min(screenImage.shape[0], screenImage.shape[1])           # find the largest possible side for the square
    print(screenImage[165][2220])


def saveCornersNumpy():

    screenImage = pag.screenshot()

    topLeft = screenImage.crop(box=(2220, 165, 2230, 175))
    topRight = screenImage.crop(box=(3028, 165, 3038, 175))
    bottomLeft = screenImage.crop(box=(2220, 975, 2230, 985))
    bottomRight = screenImage.crop(box=(3028, 975, 3038, 985))

    np.save("Corners/topLeft", np.array(topLeft))
    np.save("Corners/topRight", np.array(topRight))
    np.save("Corners/bottomLeft", np.array(bottomLeft))
    np.save("Corners/bottomRight", np.array(bottomRight))

    plt.subplot(2, 2, 1)
    plt.imshow(topLeft)
    plt.subplot(2, 2, 2)
    plt.imshow(topRight)
    plt.subplot(2, 2, 3)
    plt.imshow(bottomLeft)
    plt.subplot(2, 2, 4)
    plt.imshow(bottomRight)
    plt.show()


def locate():

    sleep(3)
    for i in range(10):
        print(pag.position())
        sleep(3)


def clearDirectory(path="Images"):

    for file in os.listdir(path):
        print(file)
        os.remove(path + "/" + file)
