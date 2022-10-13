import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


import CaptureScreen
import pyautogui as pag


class ChessBoard:

    def __init__(self):
        self.board = [["_"] * 8]*8
        self.grid = CaptureScreen.findGrid()            # Locate co-ordinates of the chessboard on screen
        # self.screenImage = np.array(pag.screenshot(region=(self.grid[1], self.grid[0], self.grid[2], self.grid[2])))
        self.screenImage = self.captureScreen()
        print(self.screenImage.shape)
        plt.imshow(self.screenImage)
        plt.show()

    def captureScreen(self):
        screenImage = pag.screenshot(region=(self.grid[1], self.grid[0], self.grid[2], self.grid[2]))
        return np.array(screenImage.resize([640, 640]))

    def updateBoard(self):
        increment = self.grid[2] // 8
        y = self.grid[0]
        x = self.grid[1]

        for j in range(8):
            for i in range(8):
                # plt.imshow(self.screenImage[y+j*increment:y+(j+1)*increment, x+i*increment:x+(i+1)*increment])
                # plt.show()
                pass


    def checkBoard(self):
        for row in self.board:
            print(row)


def gamePlay():

    board = ChessBoard()            # initiate the board
                                    # get the coordinates of where it sits on the screen
    board.checkBoard()
    board.updateBoard()
