import imp
from flask import render_template
from requests import request
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from keras.models import load_model
import numpy as np
from numpy import load
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Flatten, Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


model = keras.models.load_model('results/model_a')

#model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
#model.summary()


app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1



@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET','POST'])
def after():
    global model, vocab, inv_vocab, resnet

    file = request.files['file1']

    file.save('static/file.jpg')

    #img = cv2.imread('static/file.jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #img = cv2.resize(img, (180,180,))
    #img = np.reshape(img, (1, 180, 180, 3))
    #print(img)

    classes = [
    'Your efforts have not gone unnoticed',
    'Message decrypted- Please take action according to Plan <redacted> immediately - Thank you',
    
    ]
    model = keras.models.load_model('results/model_a')

    image = tf.keras.preprocessing.image.load_img('static/file.jpg')
    image = image.resize((180,180))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    predict = int(np.argmax(predictions, axis=1))
    final = classes[predict]

    return render_template('predict.html', final=final)

if __name__ == '__main__':
    app.run(debug=True)

