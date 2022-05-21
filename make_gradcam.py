'''
Descripttion: Store Gram-Cam Image in /4tssd/imagenet/imagenet_gradcam/
version: 
Author: QIU Yaowen
Date: 2021-10-16 21:54:37
LastEditors: Andy
LastEditTime: 2022-04-16 22:19:30
'''
import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
from modified_gradcam import *
import config
from tensorflow.keras.applications.resnet50 import preprocess_input
import json
import pandas as pd
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


imagenet_path = config.train

gramcam_path = "/4tssd/imagenet/full_gradcam/"


if not os.path.exists(gramcam_path):
    os.makedirs(gramcam_path)

def process(x):
        # print(x.shape)
    i = tf.cast(x, dtype = tf.uint8)
    x = tf.cast(i, tf.float32)
    # x = tf.keras.applications.inception_v3.preprocess_input(x)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    return x

# model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')
model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

grad_model = keras.models.Model(
    [model.inputs], [model.get_layer("conv5_block3_out").output, model.output]
)
for layer in grad_model.layers:
    layer.trainable = False

# Class json file
class_labels = json.load(open('CWOX/imagenet_class_index.json', 'r'))

# folder : nums_label
folder_to_nums = {class_labels[key][0]: key for key in class_labels.keys()}


def transform_gradcam(img_path, grad_model, pred_index,cam_path):

    processed_img_array = process(get_img_array(img_path, size=(224,224)))

    heatmap = make_gradcam_heatmap(processed_img_array, grad_model,pred_index)

    save_and_display_gradcam(img_path, heatmap,cam_path)


for folder in os.listdir(imagenet_path):

    if not os.path.exists(gramcam_path+folder):
        os.makedirs(gramcam_path+folder)

    pred_index = int(folder_to_nums[folder])
    
    for img in os.listdir(imagenet_path+folder+'/'):

        if os.path.exists(gramcam_path+folder+'/'+img): #if exists such file
            continue

        img_path = imagenet_path+folder+'/'+img

        transform_gradcam(img_path,grad_model,pred_index,gramcam_path+folder+'/'+img)

    print(folder + "SUCCESS!")
