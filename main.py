import CaptureScreen
import IdentifyChessPieces
import ReadImageDemo
import BuildImageDataset


def main():

    pass
    # CaptureScreen.capture()
    # ReadImage.buildModel()                    # build machine model and save
    # ReadImageDemo.loadModel()                 # load model and make predictions
    # IdentifyChessPieces
    BuildImageDataset.createDataset()         # create a numpy array of images
    BuildImageDataset.loadDataset()             # load the numpy array, probably remove in future


if __name__ == '__main__':
    main()
