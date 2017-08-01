import numpy as np


class Plot:

    @staticmethod
    def history(hobject, savepath=None):
        hh = hobject.history
        tcost, tacc = hh["loss"], hh["acc"]
        vcost, vacc = hh["val_loss"], hh["val_acc"]
        Plot._common(tcost, tacc, vcost, vacc, savepath=savepath)

    @staticmethod
    def _common(tcost, tacc, vcost, vacc, savepath=None):
        from matplotlib import pyplot

        epochs = np.arange(1, len(tcost) + 1)
        fig, (axt, axb) = pyplot.subplots(2, 1, figsize=(8, 7))
        axt.plot(epochs, tcost, "b-", label="Testing")
        axt.plot(epochs, vcost, "r-", label="Validation")
        axb.plot(epochs, tacc, "b-", label="Testing")
        axb.plot(epochs, vacc, "r-", label="Validation")
        axt.set_title("Cost")
        axb.set_title("Accuracy")
        axt.legend()
        axb.legend()
        pyplot.title("Learning dynamics")
        pyplot.tight_layout()
        pyplot.show()

    @staticmethod
    def architecture(model, outputpath):
        from keras.utils import plot_model
        plot_model(model, outputpath, show_shapes=True, show_layer_names=True)
