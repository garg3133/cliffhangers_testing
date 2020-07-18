import keras
import cv2
import tensorflow as tf
from keras.applications import MobileNet            
from keras.applications.imagenet_utils import decode_predictions
model = MobileNet(weights='imagenet')                                    

from django.conf import settings

def predict(image):
    path = settings.MEDIA_ROOT + '/' + image
    print(path)
    img = cv2.imread(path)  # Path of image instead of file name inside " "
    img = cv2.resize(img,(299,299))

    img = tf.keras.backend.expand_dims(
        img, axis=0
    )
    preds = model.predict(img)
    print('Predicted:', decode_predictions(preds, top=10))
    return decode_predictions(preds, top=10)
