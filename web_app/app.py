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
from services.UIEvaluatorService import preprocess_image, model

app = Flask(__name__)


@app.route(f'/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route(f'/api/predict/', methods=['POST'])
def predict():
    try:
        imagefile = request.files.get('file', '')
        print(':: Got the image from request')
        pill_img = Image.open(imagefile)
        print(':: Got the image as Pillow Object')
        preprocessed_img = preprocess_image(pill_img)
        print(':: Preprocessed the image to feed into predict method')

        prediction = model.predict(preprocessed_img)
        print(':: Got Predicted results')

        return {
            'appealing_contrast': json.dumps(prediction[0][0].item()),
            'minimalist': json.dumps(prediction[0][1].item()),
            'visual_load': json.dumps(prediction[0][2].item())
        }
    except Exception as err:
        print('Error : ' + str(err))
        print(':: Return empty results')

    return {
        'appealing_contrast': None,
        'minimalist': None,
        'visual_load': None
    }


if __name__ == '__main__':
    app.run(debug=True)
