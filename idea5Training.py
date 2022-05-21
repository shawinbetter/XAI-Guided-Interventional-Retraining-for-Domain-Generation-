'''
Descripttion: 
version: 
Author: QIU Yaowen
Date: 2022-04-21 23:04:39
LastEditors: Andy
LastEditTime: 2022-04-26 21:33:33
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
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Input,Lambda,Flatten
from tensorflow.keras.regularizers import l2



if __name__ == '__main__':
    # log_name = 'GradCamTraining_{}_{}_{}_{}.txt'.format(config.delta1_frac,config.delta2_frac,config.C_PERCENTILE,config.X_THRESHOLD)
    log_name = 'Idea5.txt'
    
    # train_path = "/4tssd/imagenet/imagenet_modified_gradcam_{}_{}_{}/".format(config.delta1_frac,config.C_PERCENTILE,config.X_THRESHOLD)
    train_path = "/4tssd/imagenet/full_gradcam/"

    optimizer = SGD(config.learning_rate,momentum=0.9)
 
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

    for layer in model.layers[0:-1]:
        layer.trainable = False

    # weight,bias = model.layers[-1].get_weights() #weight & bias for last dense layer

     #bias and weight fixer
    # def weight_initializer(shape, dtype=None):
        # return weight

    # def bias_initializer(shape, dtype=None):
        # return bias

    # Check the trainable status of the individual layers
    print("*****CHECKING TRAINABLE FOR BASE MODEL******")
    for layer in model.layers:
        print(layer.trainable)


    #define IR model
    x = model.layers[-2].output
    # x = Lambda(intervene)(x)
    # y = Dense(1000,kernel_initializer = weight_initializer,bias_initializer = bias_initializer,
                    # kernel_regularizer=l2(config.weight_decay), bias_regularizer=l2(config.weight_decay),activation='softmax')(x) #weight decay
    
    y = Dense(1000,activation='softmax')(x) #weight decay

    IR_model = Model(inputs = model.input, outputs = y)
                            
    for layer in IR_model.layers[0:-2]:
        layer.trainable = False
    
    print(IR_model.summary())
    
    print("*****CHECKING TRAINABLE FOR IR MODEL******")
    for layer in IR_model.layers:
        print(layer.trainable)

    IR_model.compile(loss='categorical_crossentropy', metrics=[
                  'accuracy', 'top_k_categorical_accuracy'], optimizer=optimizer)

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
    # csv_logger = CSVLogger('logs/GradCamTraining_full.log'.format(config.delta1_frac,config.delta2_frac,config.C_PERCENTILE,config.X_THRESHOLD))
    csv_logger = CSVLogger('logs/GradCamTraining_full.log')

    history = IR_model.fit(x=train_generator,
                        epochs=config.epochs, verbose=1,
                        max_queue_size=config.max_queue_size,
                        workers=config.workers,
                        callbacks=[csv_logger])


    ######## Save the final Model #############
    w,b = IR_model.layers[-1].get_weights()
    # np.save('model/GradCamTraining_{}_{}_{}_{}.npy'.format(config.delta1_frac,config.delta2_frac,config.C_PERCENTILE,config.X_THRESHOLD),(w,b))
    np.save('model/GradCamTraining_full.npy',(w,b))