'''
Descripttion: 
version: 
Author: QIU Yaowen
Date: 2022-03-20 20:45:58
LastEditors: Andy
LastEditTime: 2022-03-21 10:03:31
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "2"
import numpy as np
import config
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy


if __name__ == '__main__':

    ########Define Fixed Parameters###############
    epochs = 5
    val_batch_size = 64
    input_shape = (224,224)
    nums_of_classes = 1000
    optimizer = Adam(lr=1e-6)

    model = ResNet50()
    model.compile(loss='categorical_crossentropy', metrics=['acc',TopKCategoricalAccuracy(k=5)],optimizer=optimizer)
    

    ########Define Model Callback#################
    filepath="Best_model_{epoch:02d}_{acc:.4f}.hdf5"

    checkpoint = ModelCheckpoint(filepath = 'model/'+filepath, monitor='acc',verbose=1,save_best_only=False)
    
    def step_decay(epoch):
        initial_lrate = 1e-6
        drop = 0.95
        epochs_drop = 1
        lrate = initial_lrate * np.power(drop,  
            np.floor((1+epoch)/epochs_drop))
        return lrate


    lr_scheduler = LearningRateScheduler(step_decay)

    csv_logger = CSVLogger('logs/baseline.log')

    #######Training############

    val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

    val_generator = val_datagen.flow_from_directory(
                                    config.val,
                                    target_size=input_shape,
                                    batch_size=val_batch_size,
                                    interpolation='bilinear')

    history = model.fit(x = val_generator, 
            validation_data = val_generator,
            epochs=epochs, verbose=1,max_queue_size=40,
            workers=20,
            callbacks=[checkpoint,lr_scheduler, csv_logger])


    ########Save the final Model#############
    model.save('model/Baseline_Model.h5')