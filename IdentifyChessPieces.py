import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers, models, optimizers
import tensorflow.keras.utils
from tensorflow.keras.optimizers import Adam
from BuildImageDataset import labelDic, revLabelDic


def build_model():
    # Load data
    training_images = np.load("training_images.npy")
    training_labels = np.load("training_labels.npy")
    testing_images = np.load("testing_images.npy")
    testing_labels = np.load("testing_labels.npy")

    # Normalize pixel values to be between 0 and 1
    training_images, testing_images = training_images / 255.0, testing_images / 255.0

    # Convert labels to one-hot encoding
    training_labels = tensorflow.keras.utils.to_categorical(training_labels)
    testing_labels = tensorflow.keras.utils.to_categorical(testing_labels)

    model = models.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(80, 80, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(96, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(104, activation="relu"))
    model.add(layers.Dense(13, activation="softmax"))

    # Compile the model
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())

    # Train the model
    model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

    # Evaluate the model
    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print("Loss: {}".format(loss))
    print("Accuracy: {}".format(accuracy))

    # Save the model
    model.save("chess_piece_classifier.h5")


def displayModel():
    identify = models.load_model("chess_piece_classifier.model")

    testing_images = np.load("testing_images.npy")
    print(testing_images.shape)

    for num in range(testing_images.shape[0]):
        prediction = identify.predict(np.array([testing_images[num]]))
        index = np.argmax(prediction)

        # show the image
        plt.subplot(7, 4, num+1)
        plt.imshow(testing_images[num])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(revLabelDic(index))

    plt.show()
