'''
Descripttion: 
version: 
Author: QIU Yaowen
Date: 2022-05-07 22:36:01
LastEditors: Andy
LastEditTime: 2022-05-08 22:44:39
'''

import config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transform import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Input,Lambda,Flatten
from tensorflow.keras.regularizers import l2



if __name__ == '__main__':
    log_name = 'SuperClassTraining.txt'
    
    train_path = "/4tssd/imagenet/superclass/"

    optimizer = Adam(config.learning_rate)
 
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

    for layer in model.layers[0:-1]:
        layer.trainable = False


    #define Binary Classification model
    x = model.layers[-2].output
    y = Dense(2,activation='softmax')(x) #binary classifier

    IR_model = Model(inputs = model.input, outputs = y)
                            
    for layer in IR_model.layers[0:-2]:
        layer.trainable = False
    
    print(IR_model.summary())
    
    print("*****CHECKING TRAINABLE FOR IR MODEL******")
    for layer in IR_model.layers:
        print(layer.trainable)

    IR_model.compile(loss='categorical_crossentropy', metrics=[
                  'accuracy',tf.keras.metrics.AUC()], optimizer=optimizer)

    ######### Define Data Generator ##############
    def process(x):
        # print(x.shape)
        i = tf.cast(x, dtype = tf.uint8)
        x = tf.cast(i, tf.float32)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        return x

    IDG = ImageDataGenerator(preprocessing_function=process)

    train_generator = IDG.flow_from_directory(directory=train_path,
                                              target_size=(224,224),
                                              batch_size=config.batch_size,
                                              interpolation='bilinear',
                                              shuffle = True)

    ########### Training ############
    csv_logger = CSVLogger('logs/SuperClassTraining.log')

    history = IR_model.fit(x=train_generator,
                        epochs=config.epochs, verbose=1,
                        class_weight = {0:0.01,1:0.99},
                        max_queue_size=config.max_queue_size,
                        workers=config.workers,
                        callbacks=[csv_logger])


    ######## Save the final Model #############
    IR_model.save("model/SuperClass.h5")