import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Sequential, optimizers

import os
import random

# settings for reproducibility
seed = 42
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# Utility function for obtaining of the errors
def obtain_errors(val_generator, predictions):
    # Get the filenames from the generator
    fnames = validation_generator.filenames

    # Get the ground truth from generator
    ground_truth = validation_generator.classes

    # Get the dictionary of classes
    label2index = validation_generator.class_indices

    # Obtain the list of the classes
    idx2label = list(label2index.keys())
    print("The list of classes: ", idx2label)

    # Get the class index
    predicted_classes = np.argmax(predictions, axis=1)

    errors = np.where(predicted_classes != ground_truth)[0]
    print("Number of errors = {}/{}".format(len(errors), validation_generator.samples))

    return idx2label, errors, fnames

# Utility function for visualization of the errors
def show_errors(idx2label, errors, predictions, fnames):
    # Show the errors
    for i in range(len(errors)):
        pred_class = np.argmax(predictions[errors[i]])
        pred_label = idx2label[pred_class]

        title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            predictions[errors[i]][pred_class])

        original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
        plt.figure(figsize=[7,7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(original)
        plt.show()


# Utility function for plotting of the model results
def visualize_results(history):
    # Plot the accuracy and loss curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def create_model(num_trainable_layers=None):
    vgg_conv = vgg16.VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(224, 224, 3))
    if num_trainable_layers is None:
        # Freeze all the layers
        for layer in vgg_conv.layers[:]:
            layer.trainable = False
    else:
        # Freeze all the layers
        for layer in vgg_conv.layers[:-num_trainable_layers]:
            layer.trainable = False

    # Create the model
    model = Sequential()
    model.add(vgg_conv)
    model.add(Flatten())
    # model.add(Dense(512, activation='relu', input_dim=7 * 7 * 512))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(3, activation='softmax'))


    return model

if __name__ == '__main__':

    """
    vgg_transfer_model.h5
    Epoch 00020: val_accuracy did not improve from 0.93333
    30/30 [==============================] - 70s 2s/step - loss: 0.2489 - accuracy: 0.9067 - val_loss: 0.3021 - val_accuracy: 0.9133
    8/7 [================================] - 12s 2s/step
    The list of classes:  ['pumpkin', 'tomato', 'watermelon']
    Number of errors = 13/150

    vgg_transfer_finetune_model.h5
    Epoch 00020: val_accuracy did not improve from 0.96667
    30/30 [==============================] - 78s 3s/step - loss: 0.0962 - accuracy: 0.9817 - val_loss: 1.1951 - val_accuracy: 0.9267
    8/7 [================================] - 11s 1s/step
    The list of classes:  ['pumpkin', 'tomato', 'watermelon']
    Number of errors = 11/150
    
    """
    model_name = 'vgg_transfer_finetune_model.h5'  #'vgg_fine_tune.h5'  #'vgg_transfer_model.h5'

    # each folder contains three subfolders in accordance with the number of classes
    train_dir = './clean-dataset/train'
    validation_dir = './clean-dataset/validation'

    # the number of images for train and test is divided into 80:20 ratio
    nTrain = 600
    nVal = 150

    # load the normalized images
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # define the batch size
    batch_size = 20

    # the defined shape is equal to the network output tensor shape
    train_features = np.zeros(shape=(nTrain, 7, 7, 512))
    train_labels = np.zeros(shape=(nTrain, 3))

    # generate batches of train images and labels
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    # generate batches of validation images and labels
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    if not Path(f"./{model_name}").exists():
        model = create_model(num_trainable_layers=4)

        # configure the model for training
        model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        # use the train and validation feature vectors
        history = model.fit(train_generator,
                            steps_per_epoch=train_generator.samples / train_generator.batch_size,
                            epochs=20,
                            validation_data=validation_generator,
                            validation_steps=validation_generator.samples / validation_generator.batch_size,
                            verbose=1,
                            callbacks=callbacks_list
                            )
        # Run the function to illustrate accuracy and loss
        visualize_results(history)

        # Save the model
        model.save(model_name)
    else:
        model = load_model(model_name)


    # Get the predictions from the model using the generator
    predictions = model.predict(validation_generator,
                                steps=validation_generator.samples / validation_generator.batch_size, verbose=1)

    # get the list of the corresponding classes
    ground_truth = validation_generator.classes

    # Run the function to get the list of classes and errors
    idx2label, errors, fnames = obtain_errors(validation_generator, predictions)

    # Run the function to illustrate the error cases
    show_errors(idx2label, errors, predictions, fnames)

