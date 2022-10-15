import datetime

import numpy as np
from tensorflow.python.keras import models
from time import sleep

import CaptureScreen
import pyautogui as pag


def indexPiece(num):
    if num == 0:
        return "wp"
    elif num == 1:
        return "bp"
    elif num == 2:
        return "wN"
    elif num == 3:
        return "bN"
    elif num == 4:
        return "wB"
    elif num == 5:
        return "bB"
    elif num == 6:
        return "wR"
    elif num == 7:
        return "bR"
    elif num == 8:
        return "wQ"
    elif num == 9:
        return "bQ"
    elif num == 10:
        return "wK"
    elif num == 11:
        return "bK"
    else:
        return "__"


class ChessBoard:

    def __init__(self):
        self.board = [['__' for i in range(8)] for i in range(8)]
        self.grid = CaptureScreen.findGrid()  # Locate co-ordinates of the chessboard on screen
        self.screenImage = None
        self.update = False

    def captureScreen(self):  # return the screen in 640X640 pixels in numpy array
        # print("CAPTURE!")
        timeFormat = '%y%m%d%H%M%S'
        screenImage = pag.screenshot(region=(self.grid[1], self.grid[0], self.grid[2], self.grid[2]))
        screenImage.save("Images/ss{}.png".format(datetime.datetime.now().strftime(timeFormat)))
        self.screenImage = np.array(screenImage.resize([640, 640]))

    def updateBoard(self):
        # print("UPDATE")
        identify = models.load_model("chess_piece_classifier.model")
        for i in range(8):
            for j in range(8):
                subImage = self.screenImage[i * 80:(i + 1) * 80, j * 80:(j + 1) * 80]
                prediction = identify.predict(np.array([subImage]))
                piece = indexPiece((np.argmax(prediction)))
                if piece != self.board[i][j] and not self.update:
                    self.update = True
                    # print("CHANGE")
                self.board[i][j] = piece

    def checkBoard(self):
        # print("Check {}".format(self.update))
        if self.update:
            for row in self.board:
                print(row)
            print()
            self.update = False


def gamePlay():
    board = ChessBoard()  # initiate the board
    # sleep(3)
    print("START")

    while True:
        board.captureScreen()
        board.updateBoard()
        board.checkBoard()
        sleep(1)

