import tensorflow as tf
from keras.layers import LSTM, Bidirectional, Dense, Layer, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanSquaredError


class LogMagSpect(Layer):
    def __init__(self, length=320, step=160):
        super(LogMagSpect, self).__init__()
        self.length = length
        self.step = step

    def log_mag(self, sig):
        z = tf.signal.stft(sig, frame_length=self.length, frame_step=self.step, fft_length=320)
        z = tf.abs(z)
        z = tf.math.log(z)
        return z

    def call(self, sigs, **kwargs):
        x = sigs[0]  # ref sig
        y = sigs[1]  # mic sig
        x_spect = self.log_mag(x)
        y_spect = self.log_mag(y)
        spect = tf.concat([x_spect, y_spect], -1)
        return spect


class BLSTM(Model):
    def __init__(self):
        super(BLSTM, self).__init__()

        self.log_mag = LogMagSpect()
        self.reshape = Reshape((1, 322))
        self.blstm1 = Bidirectional(LSTM(300, return_sequences=True))
        self.blstm2 = Bidirectional(LSTM(300, return_sequences=True))
        self.blstm3 = Bidirectional(LSTM(300, return_sequences=True))
        self.blstm4 = Bidirectional(LSTM(300, return_sequences=True))
        self.dense_out = Dense(161, activation='sigmoid')

    def call(self, sig, training=None, mask=None):
        x = self.log_mag(sig)
        x = self.reshape(x)
        x = self.blstm1(x)
        x = self.blstm2(x)
        x = self.blstm3(x)
        x = self.blstm4(x)
        x = self.dense_out(x)
        return x


if __name__ == "__main__":
    lr = 0.001
    model = BLSTM()

    model.compile(optimizer=Adam(learning_rate=lr), loss=MeanSquaredError(), metrics=['accuracy'])
    model.build(input_shape=(2, 320, ))
    print(model.summary())
