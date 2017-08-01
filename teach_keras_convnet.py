from util import pull_data

from keras.models import Sequential
from keras.layers import (
    Conv1D, MaxPool1D, Flatten, Dense, BatchNormalization, Activation
)


def build_keras_cnn(inshape, outshape):

    def convblock(nf, fx):
        return [BatchNormalization(),
                Conv1D(filters=nf, kernel_size=fx, data_format="channels_first"),
                MaxPool1D(2, data_format="channels_first"),
                Activation("relu")]

    model = Sequential(
        layers=[BatchNormalization(input_shape=inshape),
                Conv1D(filters=12, kernel_size=3),
                MaxPool1D(2),
                Activation("relu")]  # -> 12x24
        + convblock(nf=12, fx=3)  # -> 12x11 = 132
        + [BatchNormalization(), Flatten(), Dense(60, activation="tanh")]
        + [Dense(outshape, activation="sigmoid")]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    return model

tX, tY, vX, vY = pull_data()

net = build_keras_cnn(inshape=tX.shape[1:], outshape=1)
net.fit(tX, tY, batch_size=10, epochs=30, validation_data=(vX, vY))
