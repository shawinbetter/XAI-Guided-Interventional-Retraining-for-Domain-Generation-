'''
Descripttion: 
version: 
Author: QIU Yaowen
Date: 2022-03-23 19:49:00
LastEditors: Andy
LastEditTime: 2022-03-26 21:30:00
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0"
import config
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


if __name__ == '__main__':

    ########Define Fixed Parameters###############
    epochs = 5
    train_batch_size = config.batch_size
    val_batch_size = 1
    input_shape = config.input_shape
    optimizer = Adam(lr = config.learning_rate)

    ########Define Model#################
    model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')


    model.compile(loss='categorical_crossentropy', metrics=['acc',TopKCategoricalAccuracy(k=5)],optimizer=optimizer)
    

    csv_logger = CSVLogger('logs/validate.log')

    #######Training############
    def process(x):
        i = tf.keras.layers.Input(x, dtype = tf.uint8)
        x = tf.cast(i, tf.float32)
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        return x
        
    train_datagen = ImageDataGenerator(process)
    val_datagen = ImageDataGenerator(process)

    train_generator = train_datagen.flow_from_directory(
                                    config.train,
                                    target_size=input_shape,
                                    batch_size=train_batch_size,
                                    interpolation='bilinear',
                                    shuffle = True)
    val_generator = val_datagen.flow_from_directory(
                                    config.val,
                                    target_size=input_shape,
                                    batch_size=val_batch_size,
                                    interpolation='bilinear',
                                    shuffle = True)

    history = model.fit(x = train_generator, 
            validation_data = val_generator, 
            epochs=epochs, verbose=1,max_queue_size=40,
            workers=20,
            callbacks=[csv_logger])


    ########Save the final Model#############
    model.save('model/validate.h5')