import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

EPOCHS = 30
BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 10000


class BOS_MLP:
    def __init__(self):
        pass

    def run(self, state, action):
        tf.random.set_seed(123)
        data_x, data_y = state

        train_x, train_y, test_x, test_y = (
            data_x[action == 1],
            data_y[action == 1].ravel(),
            data_x[action == 0],
            data_y[action == 0].ravel(),
        )
        

        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).cache()
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(BATCH_SIZE)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1),
            ]
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3,
            mode='min',
            restore_best_weights=True)

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.MeanSquaredError())
        model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, callbacks = [early_stopping], verbose=0)

        return model.predict(test_ds)
