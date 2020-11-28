import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img


class_target_names = ['pumpkin', 'tomato', 'watermelon']

if __name__ == '__main__':

    # read in image
    img_path = '/Users/patrickryan/Development/python/mygithub/learnopencv/Keras-Transfer-Learning/test-dataset/pumpkin/pumpkin3.jpg'
    img = load_img(img_path,
                   target_size=(224,224))

    # scale image
    img = img_to_array(img) / 255.0
    print(img.shape)

    # read in model
    model = load_model('vgg_transfer_model.h5')

    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = np.argmax(pred, axis=-1)

    print(pred, pred_class, class_target_names[pred_class[0]])
