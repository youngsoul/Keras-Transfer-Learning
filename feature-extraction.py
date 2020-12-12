import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img


import os
import random

if __name__ == '__main__':
    # settings for reproducibility
    seed = 42
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    vgg_conv = vgg16.VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(224, 224, 3))

    # VGG16 layer by layer
    vgg_conv.summary()

    # each folder contains three subfolders in accordance with the number of classes
    train_dir = './clean-dataset/train'
    validation_dir = './clean-dataset/validation'

    # the number of images for train and test is divided into 80:20 ratio
    nTrain = 600
    nVal = 150

    # load the normalized images
    datagen = ImageDataGenerator(rescale=1. / 255)

    # define the batch size
    batch_size = 20

    # the defined shape is equal to the network output tensor shape
    train_features = np.zeros(shape=(nTrain, 7, 7, 512))
    train_labels = np.zeros(shape=(nTrain, 3))

    # generate batches of train images and labels
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    # iterate through the batches of train images and labels
    for i, (inputs_batch, labels_batch) in enumerate(train_generator):
        if i * batch_size >= nTrain:
            break
            # pass the images through the network
        features_batch = vgg_conv.predict(inputs_batch)
        train_features[i * batch_size: (i + 1) * batch_size] = features_batch
        train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch

    # reshape train_features into vector
    train_features_vec = np.reshape(train_features, (nTrain, 7 * 7 * 512))
    print("Train features Vec Shape: {}".format(train_features_vec.shape))
    np.save('training-features.npy', np.asarray(train_features_vec))
    np.save('training-labels.npy', np.asarray(train_labels))

    # --------------------------------------------------------------
    # validation
    validation_features = np.zeros(shape=(nVal, 7, 7, 512))
    validation_labels = np.zeros(shape=(nVal, 3))

    # generate batches of validation images and labels
    validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    # iterate through the batches of validation images and labels
    for i, (inputs_batch, labels_batch) in enumerate(validation_generator):
        if i * batch_size >= nVal:
            break
        features_batch = vgg_conv.predict(inputs_batch)
        validation_features[i * batch_size: (i + 1) * batch_size] = features_batch
        validation_labels[i * batch_size: (i + 1) * batch_size] = labels_batch

    # reshape validation_features into vector
    validation_features_vec = np.reshape(validation_features, (nVal, 7 * 7 * 512))
    print("Validation features: {}".format(validation_features_vec.shape))
    np.save('val-features.npy', np.asarray(validation_features_vec))
    np.save('val-labels.npy', np.asarray(validation_labels))


