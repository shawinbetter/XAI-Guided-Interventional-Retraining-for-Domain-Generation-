'''
Descripttion: 
version: 
Author: QIU Yaowen
Date: 2022-04-21 22:48:08
LastEditors: Andy
LastEditTime: 2022-05-08 12:19:36
'''


import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from modified_gradcam import *
import config
import json
import pandas as pd
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


imagenet_path = config.train

model = tf.keras.models.load_model("model/SuperClass.h5")

model.compile(loss='categorical_crossentropy', metrics=['acc'])

grad_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer("conv5_block3_out").output, model.output]
)

w = model._layers[-1].weights[0].numpy()

last_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer("avg_pool").output]
)
# Class json file
class_labels = json.load(open('CWOX/imagenet_class_index.json', 'r'))

# folder : nums_label
folder_to_nums = {class_labels[key][0]: key for key in class_labels.keys()}

#######Process############
def process(x):
    # print(x.shape)
    i = tf.cast(x, dtype = tf.uint8)
    x = tf.cast(i, tf.float32)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = tf.expand_dims(x,axis = 0)
    return x

def make_gradcam_heatmap(img_array, grad_model, pred_index):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # if pred_index is None:
        #     pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path):

    # plt.figure(figsize=[12,8],dpi=200)
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path,target_size=(224,224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    jet_heatmap = jet_heatmap / 255
    
    threshold = np.max(jet_heatmap) * config.delta1_frac
    
    img[jet_heatmap[:,:,0] < threshold] = 0
    
    return img

def transform_gradcam(img_path,processed_img_array, grad_model, pred_index,cam_path):
    
    heatmap = make_gradcam_heatmap(processed_img_array, grad_model,pred_index)

    return save_and_display_gradcam(img_path, heatmap,cam_path)


Contribution_Matrix = np.zeros((2048,2))

N = len(os.listdir("/4tssd/imagenet/superclass/car/"))

for img in os.listdir("/4tssd/imagenet/superclass/car/"):

    img_path = "/4tssd/imagenet/superclass/car/"+img

    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    except:
        continue
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    processed_array = process(img_array)

    purified_img_array = transform_gradcam(img_path,processed_array,grad_model,0,None)

    h_prime = last_model(process(purified_img_array)).numpy().T

    contribution = np.multiply(h_prime,w)

    Contribution_Matrix += contribution

Contribution_Matrix /= N

np.save("model/contribution_car.npy",Contribution_Matrix)

