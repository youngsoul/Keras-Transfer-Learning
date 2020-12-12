import numpy as np


from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential, optimizers
from matplotlib import pyplot


def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['acc'], color='blue', label='train')
    pyplot.plot(history.history['val_acc'], color='orange', label='test')
    # save plot to file
    filename = 'feature-extract-history'
    pyplot.savefig(filename + '-plot.png')
    pyplot.close()


if __name__ == '__main__':
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=7 * 7 * 512))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    # configure the model for training
    model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    train_features_vec = np.load('training-features.npy')
    train_labels = np.load('training-labels.npy')

    validation_features_vec = np.load('val-features.npy')
    validation_labels = np.load('val-labels.npy')

    # define the batch size
    batch_size = 20

    # use the train and validation feature vectors
    history = model.fit(train_features_vec,
                        train_labels,
                        epochs=20,
                        batch_size=batch_size,
                        validation_data=(validation_features_vec, validation_labels))
    print(history.history.keys())
    summarize_diagnostics(history)
