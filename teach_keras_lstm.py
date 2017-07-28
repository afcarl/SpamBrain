from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Dense, Bidirectional

from util import pull_data


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


X, Y = pull_data()
X = X[..., 0].transpose(0, 2, 1)

net = build_keras_rnn(inshape=X.shape[1:], outshape=1)
net.fit(X, Y, epochs=30, validation_split=0.1)
