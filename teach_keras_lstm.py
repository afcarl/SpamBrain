from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Dense, Bidirectional

from util import pull_data, projectroot
from create_plots import Plot


def build_keras_rnn(inshape, outshape):
    model = Sequential(layers=[
        BatchNormalization(input_shape=inshape),
        Bidirectional(LSTM(180, activation="relu", return_sequences=True, implementation=0)),
        Bidirectional(LSTM(120, activation="relu", return_sequences=True, implementation=0)),
        Bidirectional(LSTM(60, activation="relu", implementation=0)),
        BatchNormalization(),
        Dense(180, activation="tanh"),
        Dense(outshape, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    return model

tX, tY, vX, vY = pull_data()

net = build_keras_rnn(inshape=tX.shape[1:], outshape=1)

Plot.architecture(net, projectroot + "LSTM.dot")

net.fit(tX, tY, epochs=30, validation_data=(vX, vY))
