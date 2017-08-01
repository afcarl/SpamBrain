from keras.models import Model
from keras.layers import (
    Conv1D, MaxPool1D,
    BatchNormalization, Input, Flatten, Concatenate, Dropout,
    Dense, Activation
)
from util import pull_data, projectroot
from create_plots import Plot


def build_keras_convolutional_graph(inshape, outshape):
    inl = Input(inshape)  # 300x50
    C = [Conv1D(50, kernel_size=7)(inl),  # 50x44
         Conv1D(50, kernel_size=5)(inl),  # 50x46
         Conv1D(50, kernel_size=3)(inl)]  # 50x48
    CP = [Flatten()(MaxPool1D()(c)) for c in C]
    CA = Activation("relu")(Concatenate()(CP))  # 3450
    CC = BatchNormalization()(CA)
    FF1 = Dropout(0.5)(Dense(360, activation="tanh")(CC))
    FF2 = Dense(120, activation="tanh")(FF1)
    O = Dense(outshape, activation="sigmoid")(FF2)
    model = Model(inl, O)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    return model


tX, tY, vX, vY = pull_data()

graph = build_keras_convolutional_graph(inshape=tX.shape[1:], outshape=1)
baseline = graph.evaluate(vX, vY, verbose=0)
Plot.architecture(graph, projectroot+"convgraph_architecture.dot")

print("Baseline accuracy on {} validation points: {:.2%} (Cost: {:.4f})"
      .format(len(vX), *baseline))

history = graph.fit(tX, tY, epochs=3, validation_data=(vX, vY))
Plot.history(hobject=history)
