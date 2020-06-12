import tensorflow as tf
import numpy as np
import pandas as pd
import pdb

EPOCHS = 20
BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 10000

class MNIST_CNN:
    def __init__(self):
        pass

    def run(self, state, action):
        tf.random.set_seed(123)
        _, data_y = state
        # cnn features for cnn architecture
        data_x = np.load("./datasets/MNIST_X.npy")
        train_x, train_y, test_x, test_y = data_x[action == 1], \
                                           data_y[action == 1], \
                                           data_x[action == 0], \
                                           data_y[action == 0]

        y_train_integer = np.argmax(train_y, axis=1)
        y_test_integer = np.argmax(test_y, axis=1)

        train_ds = tf.data.Dataset.from_tensor_slices((train_x, y_train_integer)).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).cache()
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, y_test_integer)).batch(BATCH_SIZE)

        # Creating a Sequential Model and adding the layers

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )        

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3,
            mode='min',
            restore_best_weights=True)

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, callbacks = [early_stopping], verbose=0)

        return np.array(model.predict_proba(test_x, verbose=0))

