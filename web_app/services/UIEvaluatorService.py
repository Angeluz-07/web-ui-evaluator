from flask import Flask
from flask import render_template, request

import json

import keras_preprocessing.image as _image
import tensorflow as tf
from keras import backend as K
from keras.backend import set_session
from keras.models import load_model
from PIL import Image
import numpy as np

IMG_WIDTH, IMG_HEIGHT = 1024//2, 768//2

# https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def F1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def preprocess_image(
    PIL_img: Image,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH
):
    img = PIL_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
    img_tensor = _image.img_to_array(img)                   # (height, width, channels)
    img_tensor = img_tensor[..., :3]
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor


"""
    Some required tensorflow configurations
    ref : https://github.com/tensorflow/tensorflow/issues/28287
"""

model = load_model('models/ui_evaluator_100_epochs.h5', custom_objects={'F1': F1 })