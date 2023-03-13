from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/"+imagefile.filename
    imagefile.save(image_path)

    img = cv2.imread(image_path)
    img_resized = tf.image.resize(img, (256, 256))

    new_model = load_model(os.path.join('models', 'parkinsons_detection.h5'))
    yhatnew = new_model.predict(np.expand_dims(img_resized / 255, 0))

    if yhatnew > 0.5:
        prediction = 'Predicted class is Parkinsons'
    else:
        prediction = 'Predicted class is Healthy'

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
