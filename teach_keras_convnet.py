from util import pull_data

from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Activation
)


def build_keras_cnn(inshape, outshape):

    def convblock(nf, fx):
        return [BatchNormalization(),
                Conv2D(filters=nf, kernel_size=(fx, 1), data_format="channels_first"),
                MaxPool2D((2, 1), data_format="channels_first"),
                Activation("relu")]

    model = Sequential(
        layers=[BatchNormalization(input_shape=inshape),
                Conv2D(filters=12, kernel_size=(3, 1), data_format="channels_first"),
                MaxPool2D((2, 1), data_format="channels_first"),
                Activation("relu")]  # -> 12x24
        + convblock(nf=12, fx=3)  # -> 12x11 = 132
        + [BatchNormalization(), Flatten(), Dense(60, activation="tanh")]
        + [Dense(outshape, activation="sigmoid")]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    return model

X, Y = pull_data()
print("X:", X.shape, "Y:", Y.shape)

net = build_keras_cnn(inshape=X.shape[1:], outshape=1)
net.fit(X, Y, batch_size=10, epochs=30, validation_split=0.1)
