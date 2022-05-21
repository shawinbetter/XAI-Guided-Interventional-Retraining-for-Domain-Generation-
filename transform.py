'''
Descripttion: 
version: 
Author: QIU Yaowen
Date: 2022-01-05 22:53:37
LastEditors: Andy
LastEditTime: 2022-03-11 21:13:23
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0"

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

import config
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

## Define H0 matrix
model_builder = ResNet50
model = model_builder(weights="imagenet")
for layer in model.layers:
    layer.trainable = False

model_no_softmax = Model(
    [model.inputs], [model.get_layer('avg_pool').output]
)
for layer in model_no_softmax.layers:
    layer.trainable = False
# Compute fixed variables

empty = np.zeros(shape=(1,224,224,3)) 
h0 = model_no_softmax(empty)

del empty,model_builder,model,model_no_softmax


def intervene(h):
    # print("H shape:",h.shape)

    delta2 = tf.cast(tf.math.reduce_max(h),float) * config.delta2_frac
    
    h = tf.math.subtract(h,h0)#compute r

    def func(hi):
        # print("Hi shape:",hi.shape)
        
        lambi = (1/delta2) * (delta2 - tf.math.minimum(delta2,hi))
        
        res = (1.0-lambi) * hi + tf.math.multiply(lambi,tf.random.normal((2048,),dtype=tf.float32))

        return res

    return tf.map_fn(func,h)


### Test
# if __name__=='__main__':
#     model_builder = keras.applications.ResNet50
#     # Make model
#     model = model_builder(weights="imagenet")
#     model.layers[-1].activation = None
#     # Delete prediction layer
#     model_no_softmax= keras.Model(inputs=model.input, outputs=model.layers[-2].output)

#     img_path = keras.utils.get_file(
#     "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
#     )
#     img_array = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(img_path))

#     x = transform(img_array,delta1=0.01,delta2 = 1)
