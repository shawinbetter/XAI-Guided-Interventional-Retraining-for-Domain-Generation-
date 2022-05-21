'''
Descripttion: 
version: 
Author: QIU Yaowen
Date: 2022-03-11 12:36:17
LastEditors: Andy
LastEditTime: 2022-05-09 20:56:11
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from transform import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
import config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    model_path = "model/SuperClass.h5"

    model = load_model(model_path)
    # model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
    # model.compile(loss='categorical_crossentropy', metrics=[
    #               'accuracy', 'top_k_categorical_accuracy'])

    #complie
    model.compile(loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    
    def process(x):
        # print(x.shape)
        i = tf.cast(x, dtype = tf.uint8)
        x = tf.cast(i, tf.float32)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        return x

    IDG = ImageDataGenerator(preprocessing_function=process)

    val_generator = IDG.flow_from_directory(directory="/4tssd/imagenet/val_superclass/",
                                            target_size=(224,224),
                                            batch_size=1,
                                            interpolation='bilinear')

    val2_generator = IDG.flow_from_directory(directory="/4tssd/imagenet/val2_superclass/",
                                             target_size=(224,224),
                                             batch_size=1,
                                             interpolation='bilinear')

    list_of_generator = [val_generator, val2_generator]

    # with open(config.logs+log_name, 'w') as f:
    for each in list_of_generator:
        loss, acc,auc,pre,rec = model.evaluate(
            x=each, max_queue_size=config.max_queue_size, workers=config.workers)
        print(f"Loss:{loss}, Top-1 Accuracy:{acc}, AUC:{auc},Precision:{pre},Recall:{rec}")
            # f.write(
            #     f"Loss:{loss},Top-1 Accuracy:{acc1},Top-5 Accuracy:{acc5} \n")
