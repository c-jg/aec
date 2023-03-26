import os
from glob import glob

import keras.models
import tensorflow as tf
from keras.layers import LSTM, Bidirectional, Dense
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

from preprocessing import get_dataset


def get_tf_record_files(dataset_name):
    return glob(os.path.join(input_path, dataset_name, "*.tfrecords"))


def main():
    train_files = get_tf_record_files("TRAIN")
    valid_files = get_tf_record_files("VALID")

    batch_size = 64
    lr = 0.001
    epochs = 20

    train_dataset = get_dataset(train_files, batch_size=batch_size)
    valid_dataset = get_dataset(valid_files, batch_size=batch_size)

    model = keras.models.Sequential([
        keras.layers.Input(shape=(24, 322)),
        Bidirectional(LSTM(322, return_sequences=True)),
        Bidirectional(LSTM(322, return_sequences=True)),
        Bidirectional(LSTM(322, return_sequences=True)),
        Bidirectional(LSTM(322, return_sequences=True)),
        Dense(161, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss=keras.losses.MeanSquaredError(), metrics=['mse'])

    print(model.summary())

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=output_path, verbose=1, save_best_only=True)

    _ = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        callbacks=[cp_callback]
    )

    print("\n--- DONE TRAINING ---\n")

    return


if __name__ == "__main__":
    input_path = ""
    output_path = ""

    main()
