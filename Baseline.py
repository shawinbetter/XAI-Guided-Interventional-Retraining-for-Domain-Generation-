'''
Descripttion: 
version: 
Author: QIU Yaowen
Date: 2021-10-16 21:54:37
LastEditors: Andy
LastEditTime: 2022-03-27 19:49:48
'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import load_model
import config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


if __name__ == '__main__':
    log_name = 'Baseline.txt'


    model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')


    # model = load_model(config.model+'BaselineModel.hdf5')

    model.compile(loss='categorical_crossentropy', metrics=[
                  'accuracy', 'top_k_categorical_accuracy'])

    def process(x):
        # print(x.shape)
        i = tf.cast(x, dtype = tf.uint8)
        x = tf.cast(i, tf.float32)
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        return x

    IDG = ImageDataGenerator(preprocessing_function = process)



    val_generator = IDG.flow_from_directory(directory=config.val,
                                            target_size=(299, 299),
                                            batch_size=1,
                                            interpolation='bilinear',shuffle = True)

    val2_generator = IDG.flow_from_directory(directory=config.val2,
                                             target_size=(299, 299),
                                             batch_size=1,
                                             interpolation='bilinear',shuffle=True)

    list_of_generator = [val_generator, val2_generator]
    with open(config.logs+log_name, 'w') as f:
        for each in list_of_generator:
            loss, acc1, acc5 = model.evaluate(
                x=each, max_queue_size=config.max_queue_size, workers=config.workers)
            print(f"Loss:{loss}, Top-1 Accuracy:{acc1}, Top-5 Accuracy:{acc5}")
            f.write(
                f"Loss:{loss},Top-1 Accuracy:{acc1},Top-5 Accuracy:{acc5} \n")
