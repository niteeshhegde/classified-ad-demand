from matplotlib import pyplot
# plot loss during training


def visualize(history):
    pyplot.subplot(221)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='valid')
    pyplot.legend()

    pyplot.subplot(222)
    pyplot.title('acc')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='valid')
    pyplot.legend()

    pyplot.subplot(223)
    pyplot.title('f1')
    pyplot.plot(history.history['f1_score'], label='train')
    pyplot.plot(history.history['val_f1_score'], label='valid')
    pyplot.legend()
