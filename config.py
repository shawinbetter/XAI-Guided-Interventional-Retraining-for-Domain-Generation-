'''
Descripttion: 
version: 
Author: QIU Yaowen
Date: 2021-12-23 23:15:34
LastEditors: Andy
LastEditTime: 2022-05-07 22:40:46
'''

## Define path
train = '/4tssd/imagenet/train/'
val = '/4tssd/imagenet/val/'
val2 = '/4tssd/imagenet/val2/'

model = 'model/'
logs = 'logs/'

## Define Training Parameters
epochs = 5
batch_size = 64
input_shape = (224,224)
workers = 20
max_queue_size = 40
learning_rate = 1e-3
weight_decay = 1e-5



## Define XGIR Parameters
last_conv_layer_name = "conv5_block3_out"
C_PERCENTILE = 20
X_THRESHOLD = 0.3
delta1_frac = 0.3
delta2_frac = 0.5