'''
Descripttion: 
version: 
Author: QIU Yaowen
Date: 2022-03-30 20:48:27
LastEditors: Andy
LastEditTime: 2022-03-30 22:00:59
'''
'''
Descripttion: 
version: 
Author: QIU Yaowen
Date: 2022-03-23 19:49:00
LastEditors: Andy
LastEditTime: 2022-03-26 21:30:00
'''

import os
from unicodedata import category
os.environ["CUDA_VISIBLE_DEVICES"] =  "0"
import config
import tensorflow as tf
from modified_gradcam import *
import json
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


if __name__ == '__main__':

    val_path = config.val

    ########Define Model#################
    model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')

    model.compile(loss='categorical_crossentropy', metrics=['acc',TopKCategoricalAccuracy(k=5)])
    
    #######Process############
    def process(x):
        # i = tf.keras.layers.Input(x, dtype = tf.uint8)
        x = tf.cast(x, tf.float32)
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        return x
    
    # Class json file
    class_labels = json.load(open('CWOX/imagenet_class_index.json', 'r'))

    # folder : nums_label
    folder_to_nums = {class_labels[key][0]: key for key in class_labels.keys()}


    category, path = [], []

    for folder in os.listdir(val_path):

        for img in os.listdir(val_path + folder + '/'):
            
            img_path = val_path + folder + '/' + img

            pred_index = int(folder_to_nums[folder])

            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299,299))
            
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            x = process(img_array)

            y = model.predict(x)

            pred = np.argmax(y)

            # print(pred, pred_index)
            
            if pred != pred_index:
                category.append(folder)
                path.append(img_path)
        
        print(folder + " SUCCESS!")

    df = pd.DataFrame({'Category':category,'Path':path})
    df.to_csv("hard_log/wrong_prediction.csv",index = None)





    

