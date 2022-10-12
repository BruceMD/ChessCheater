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
    maxSide = min(screenImage.shape[0], screenImage.shape[1])  # find the largest possible side for the square
    minSide = 400  # minimum acceptance
    print(screenImage[165][2220] == cellColour)
    print(array_equal(screenImage[165][2220], cellColour))

    counter = 0

    for y in range(screenImage.shape[0] - 401):
        for x in range(screenImage.shape[1] - 401):
            if array_equal(screenImage[y][x], cellColour):
                if findCorner(y, x, screenImage, cellColour):
                    counter += 1
                    print("Cool, we found corner {}".format(counter))
                    side = expand(y, x, screenImage, cellColour)
                    if side >= 400:
                        if checkPerimeter(y, x, screenImage, cellColour, side):
                            print(x, y)
                            subImage = np.array(screenImage[y:y + side, x:x + side])
                            print(subImage.shape)
                            plt.imshow(subImage)
                            plt.show()
                            return y, x, side


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
