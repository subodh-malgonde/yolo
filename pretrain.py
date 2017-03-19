import numpy as np
import pandas as pd
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Convolution2D, ThresholdedReLU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

keras.backend.set_image_dim_ordering('tf')


def preprocess_image(image):
    # image = crop_and_resize(image)
    # image = image.astype(np.float32)

    #Normalize image
    image = image/255.0 - 0.5
    return image


def get_augmented_row(path):
    # label = int(row[' Label'])

    image = load_img(path.strip())
    image = img_to_array(image)

    # Crop, resize and normalize the image
    image = preprocess_image(image)
    return image


def get_data_generator(X, y, batch_size=32):
    N = X.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start + batch_size

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        for index in range(start, end):
            X_batch[j], y_batch[j] = get_augmented_row(X[index]), y[index]
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch


def get_model():
    model = Sequential()
    # model.add(Lambda(preprocess_batch, input_shape=(160, 320, 3), output_shape=(64, 64, 3)))

    # layer 1 output shape is 16x15x15
    model.add(Convolution2D(16, 5, 5, input_shape=(64, 64, 3), subsample=(1, 1), border_mode="valid", activation="relu"))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # layer 2 output shape is 16X13x13
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid", activation="relu"))
    # model.add(Dropout(.4))

    # Flatten the output
    model.add(Flatten())

    model.add(Dense(2048, activation="relu"))
    # layer 4
    model.add(Dense(1024, activation="relu"))
    # model.add(Dropout(.3))

    # layer 5
    model.add(Dense(512, activation="relu"))

    # Finally a single output, since this is a regression problem
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(0.0005)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model

if __name__ == "__main__":
    BATCH_SIZE = 32

    data_frame = pd.read_csv('data_set.csv', usecols=[0, 1])

    # shuffle the data
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    y = np.asarray(data_frame[' Label'])
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    X = np.asarray(data_frame['Path'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # release the main data_frame from memory
    data_frame = None

    training_generator = get_data_generator(X_train, y_train, batch_size=BATCH_SIZE)
    validation_data_generator = get_data_generator(X_val, y_val, batch_size=BATCH_SIZE)

    model = get_model()

    samples_per_epoch = (5000//BATCH_SIZE)*BATCH_SIZE

    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=1, nb_val_samples=2000)

    print("Saving model weights and configuration file.")

    model.save_weights('model.h5')  # always save your weights after training or during training
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())