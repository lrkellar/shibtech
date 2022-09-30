# Start point: Working Image classification with predictions function outputting,acc max %75
# Working Model, need to fix onehot class exports for use with flask app

import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.preprocessing import OneHotEncoder



# Hard paths
data = pd.read_csv('ShibV5/multielement.csv')


# Class label processing

def prep_classes(df):
    classes = []
    for i, row in df.iterrows():
        classes.append(row['label'])
    return np.unique(classes)

prepped_classes = prep_classes(data)

num_classes = len(prepped_classes)

encoder_lab = LabelEncoder()
data['label'] = encoder_lab.fit_transform(data['label'])
np.save('ShibV5/export/class_labels.npy', encoder_lab.classes_)



npz_paths = []
width = 120
height = 120
channels = 3
# ID preprocess
# integer encode
id_to_int = LabelEncoder()
id_to_int.fit(data['user_id'])
np.save('ShibV5/export/id_to_int.npy', id_to_int.classes_)
integer_encoded = id_to_int.transform(data['user_id'])
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
np.save('ShibV5/export/id_one_hot.npy', onehot_encoder.categories_)

for i, row in data.iterrows():
    picture_path = row['image_path']
    
    npz_path = picture_path.split('.')[0] + '-' + str(i) + '.npz'
    npz_path = '/home/lance/Projects/shibtech/ShibV5/data/' + npz_path.split('/')[-1]

    npz_paths.append(npz_path)
    #cv2 records colours as blue green red instead of standard rgb
    pic_bgr_arr = cv2.imread(picture_path)
    pic_bgr_arr = cv2.resize(pic_bgr_arr, (width, height))
    pic_rgb_arr = cv2.cvtColor(pic_bgr_arr, cv2.COLOR_BGR2RGB)

    print(f'Data prep step: {i}')

    user_ids = onehot_encoded[i]

    labels = row['label']

    print(user_ids, labels)

    np.savez_compressed(npz_path, pic = pic_rgb_arr, user_ids=user_ids, labels=labels)

data['NPZ_Path']= pd.Series(npz_paths)

close = np.load('/home/lance/Projects/shibtech/ShibV5/data/close.npz')

print(close['pic'])

def get_X_y(df):
    X_pic, X_stats = [],[]
    y = []

    for name in df['NPZ_Path']:
        loaded_npz = np.load(name)
        pic = loaded_npz['pic']
        X_pic.append(pic)

        stats = loaded_npz['user_ids']
        X_stats.append(stats)

        #print(f'The label for {name} is {loaded_npz["labels"]}')
        y.append(loaded_npz['labels'])

    X_pic, X_stats = np.array(X_pic), np.array(X_stats)

    y = np.array(y)

    return (X_pic, X_stats), y

(X_train_pic, X_train_stats), y_train = get_X_y(data)
(X_test_pic, X_test_stats), y_test = get_X_y(data)
(X_val_pic, X_val_stats), y_val = get_X_y(data)


# Data to test model
data_test = False
if data_test:
    fashion_mnist_data = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()

    print(train_images.shape)
    # (60000, 28, 28)
    width = 28
    height = 28
    channels = 1

# Model Definition

pic_input = layers.Input(shape=((width, height, channels)))
x = layers.Conv2D(16, 3, padding='same', activation='relu')(pic_input)
#x =  layers.MaxPooling2D()(x)
x =  layers.Conv2D(32, 3, padding='same', activation=layers.LeakyReLU(alpha=0.01))(x)
#x =  layers.MaxPooling2D()(x)
#x = layers.Dropout(0.4)(x)
x =  layers.Conv2D(64, 3, padding='same', activation=layers.LeakyReLU(alpha=0.01))(x)
#x =  layers.MaxPooling2D()(x)
#x = layers.Dropout(0.4)(x)
x =  layers.Flatten()(x)
x =  layers.Dense(128)(x)
x = layers.Dense(num_classes)(x)

input_onehot_ids = layers.Input(shape=(5,))
y = layers.Dense(64, activation='relu')(input_onehot_ids)
y = layers.Dense(64, activation=layers.LeakyReLU(alpha=0.01))(y)
y = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.01))(y)

z = layers.concatenate([x,y])
z = layers.Dense(128)(z)
z = layers.Dense(128)(z)
z = layers.Dense(128)(z)
z = layers.Dense(256)(z)

output1 = layers.Dense(num_classes)(z)


model = Model(inputs=[pic_input, input_onehot_ids], outputs=output1)


# Model Assembly


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training


epochs=200

if data_test == True:
    history = model.fit(    
    x = train_images, y = train_labels,
    validation_split=.2,
    epochs=epochs)
else:
    history = model.fit(
    x = [X_train_pic, X_test_stats], y = y_train,
    validation_data=([X_val_pic, X_val_stats], y_val),
    epochs=epochs
    )

model.save('ShibV5/export/test_model_2')

# Plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# error area 8-6
def prep_predict_pic(pic_path):
    pic_bgr_arr = cv2.imread(pic_path)
    pic_bgr_arr = cv2.resize(pic_bgr_arr, (width, height))
    pic_rgb_arr = cv2.cvtColor(pic_bgr_arr, cv2.COLOR_BGR2RGB)

    return np.array(pic_rgb_arr)

def prep_predict_id(id):
    id = int(id)
    id = list(id)
    id = np.array(id)

    return id

def get_test_X(picture_path, id):
    X_pic, X_stats = [],[]


    #cv2 records colours as blue green red instead of standard rgb
    pic_bgr_arr = cv2.imread(picture_path)
    pic_bgr_arr = cv2.resize(pic_bgr_arr, (120, 120))
    pic_rgb_arr = cv2.cvtColor(pic_bgr_arr, cv2.COLOR_BGR2RGB)
    X_pic.append(pic_rgb_arr)

    int_id = id_to_int.transform([id])
    id_int = int_id.reshape(len(int_id),1)
    onehot_step = onehot_encoder.transform(id_int)
    print(id)
    X_pic, X_stats = np.array(X_pic), np.array(onehot_step)

    return (X_pic, X_stats)

def make_predict(pic_path, id):
    (pt_x1, pt_x2) = get_test_X(pic_path, id)
    prediction = model.predict([pt_x1, pt_x2])
    print(prediction)
    predict = int(np.argmax(prediction, axis=1))
    final = encoder_lab.inverse_transform([predict])
    print(f'Output for User {pt_x2} is: \n {final}')

pic_path = '/home/lance/Shibboleth/Working Draft/WD-Data/static/assets/genaratedfracs/right1.png'
pic_path2 = '/home/lance/Shibboleth/Working Draft/WD-Data/static/assets/genaratedfracs/close.png'
for x in range(1,5,1):
    print(f'Loop Number: {x}')
    make_predict(pic_path=pic_path, id=x)


pt_x1, pt_x2 = get_test_X(pic_path2, 1)
print( pt_x2)
pred2 = model.predict([pt_x1,pt_x2])
predict = int(np.argmax(pred2, axis=1))
final = encoder_lab.inverse_transform([predict])
print(final, pred2)

pt_x1, pt_x2 = get_test_X(pic_path2, 4)
print( pt_x2)
pred2 = model.predict([pt_x1,pt_x2])
predict = int(np.argmax(pred2, axis=1))
final = encoder_lab.inverse_transform([predict])
print(final, pred2)