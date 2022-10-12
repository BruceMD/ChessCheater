from CaptureScreen import findGrid
import numpy as np


# instead of this nonsense global variables, need to implement OOP with these values
y, x, side, screenImage = findGrid()


def play():
    global y
    global x
    global side
    global screenImage
    print(y, x, side)
    board = [["_"] * 8]*8

    for row in board:
        print(row)


def setupBoard():
    global y
    global x
    global side
    global screenImage
