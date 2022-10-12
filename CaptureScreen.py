import os
import numpy as np
from numpy import array_equal
import pyautogui as pag
from time import sleep
import matplotlib.pyplot as plt


def findGrid():
    screenImage = np.array(pag.screenshot())
    print(screenImage.shape)

    cellColour = np.array([49, 46, 43])  # assume this grey

    for y in range(screenImage.shape[0] - 401):
        for x in range(screenImage.shape[1] - 401):
            if array_equal(screenImage[y][x], cellColour):
                if findCorner(y, x, screenImage, cellColour):
                    side = expand(y, x, screenImage, cellColour)
                    if side >= 400:
                        if checkPerimeter(y, x, screenImage, cellColour, side):
                            subImage = np.array(screenImage[y:y + side, x:x + side])
                            plt.imshow(subImage)
                            plt.show()
                            return y+1, x+1, side, screenImage


def checkPerimeter(y, x, screenImage, cellColour, side):
    for i in range(1, side):
        if not array_equal(screenImage[y+side][x+i], cellColour):
            return False
        if not array_equal(screenImage[y+i][x+side], cellColour):
            return False
    return True


def expand(y, x, screenImage, cellColour):
    s = 5
    while True:
        if array_equal(screenImage[y + s][x], cellColour) and not array_equal(screenImage[y + s - 2][x + 2],
                                                                              cellColour):
            if array_equal(screenImage[y][x + s], cellColour) and not array_equal(screenImage[y + 2][x + s - 2],
                                                                                  cellColour):
                s += 1
            else:
                return s
        else:
            return s


def findCorner(y, x, screenImage, cellColour):
    if array_equal(screenImage[y + 2][x + 2], cellColour):
        return False
    for i in range(1, 5):
        if not array_equal(screenImage[y][x + i], cellColour):
            return False
        if not array_equal(screenImage[y + 1][x], cellColour):
            return False
    return True


# a bit irrelevant at the moment
def locate():
    sleep(3)
    for i in range(10):
        print(pag.position())
        sleep(3)


def clearDirectory(path="Images"):
    for file in os.listdir(path):
        print(file)
        os.remove(path + "/" + file)
