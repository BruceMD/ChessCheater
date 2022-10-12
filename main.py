import CaptureScreen
import IdentifyChessPieces
import ReadImageDemo
import BuildImageDataset


def main():

    pass
    CaptureScreen.findGrid()
    # CaptureScreen.saveCornersNumpy()
    # CaptureScreen.locate()
    # CaptureScreen.clearDirectory()
    # ReadImageDemo.buildModel()                    # build machine model and save
    # ReadImageDemo.loadModel()                 # load model and make predictions

    # BuildImageDataset.createTrainingDataset()         # create a numpy array of images
    # BuildImageDataset.createTestingDataset()         # create a numpy array of images
    # # BuildImageDataset.loadDataset()             # load the numpy array, probably remove in future
    #
    # IdentifyChessPieces.buildModel()
    # IdentifyChessPieces.loadModel()
    # IdentifyChessPieces.displayTesting()


if __name__ == '__main__':
    main()
