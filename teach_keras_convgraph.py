from keras.models import Model
from keras.layers import (
    Conv1D, MaxPool1D,
    BatchNormalization, Input, Flatten, Concatenate,
    Dense, Activation
)

from util import pull_data


def build_keras_convolutional_graph(inshape, outshape):
    inl = Input(inshape)  # 300 x 50
    C = [Conv1D(50, kernel_size=7)(inl),  # 50x44
         Conv1D(50, kernel_size=5)(inl),  # 50x46
         Conv1D(50, kernel_size=5)(inl)]  # 50x48
    CP = [Flatten()(MaxPool1D()(c)) for c in C]
    CA = Activation("relu")(Concatenate()(CP))  # 3450
    CC = BatchNormalization()(CA)
    FF1 = Dense(360, activation="tanh")(CC)
    FF2 = Dense(120, activation="tanh")(FF1)
    O = Dense(outshape, activation="sigmoid")(FF2)
    model = Model(inl, O)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    return model


X, Y = pull_data()
X = X[..., 0].transpose(0, 2, 1)

print("X:", X.shape, "Y:", Y.shape)

graph = build_keras_convolutional_graph(inshape=X.shape[1:], outshape=1)
graph.fit(X, Y, batch_size=10, epochs=30, validation_split=0.1)
