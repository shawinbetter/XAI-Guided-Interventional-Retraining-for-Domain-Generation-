'''
Descripttion: Store cluster heatmap in /4tssd/imagenet/imagenet_cluster_heatmap/
version: 
Author: QIU Yaowen
Date: 2021-10-16 21:54:37
LastEditors: Andy
LastEditTime: 2022-02-25 15:16:36
'''

import json
import numpy as np
import torch
from torchvision import models, transforms
from scipy.special import softmax
import matplotlib.pyplot as plt
from PIL import Image
# import multiprocessing
from CWOX.plt_wox import imsc
from CWOX.apply_hltm import *
from torchray.attribution.grad_cam import grad_cam
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

is_gpu = torch.cuda.is_available()
gpu_nums = torch.cuda.device_count()
gpu_index = torch.cuda.current_device()
print("Has GPUs :", is_gpu)
print("Nums of GPUs :", gpu_nums)
print("GPU Index :", gpu_index)

device_name = torch.cuda.get_device_name(gpu_index)
print("Device Name :", device_name)


# Define Path
imagenet_path = '/4tssd/imagenet/train/'
gramcam_path = '/4tssd/imagenet/imagenet_cluster_heatmap/'

if not os.path.exists(gramcam_path):
    os.makedirs(gramcam_path)

# Load class labels
class_labels = json.load(open('CWOX/imagenet_class_index.json', 'r'))

# folder : nums_label
folder_to_nums = {class_labels[key][0]: key for key in class_labels.keys()}

## Loader and Model
loader = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
model = models.resnet50(pretrained=True).cuda()
_ = model.train(False)  # put model in evaluation mode

# Load Cluster
clusterify_resnet50 = apply_hltm(cut_level=0, json_path="CWOX/ResNet50.json")


# Define function to get heatmap for cluster
def BP_logP(class_use, image1, layer_use, model):
    poss = class_use
    probs = model(image1.cuda()).softmax(-1)
    grad = torch.zeros_like(probs)
    poss_p = probs[0, poss]
    grad[0, poss] = poss_p / poss_p.sum()
    saliency = grad_cam(model, image1.cuda(), grad,
                        saliency_layer=layer_use, resize=True)[0, 0]
    saliency = saliency.data.cpu().numpy()

    return saliency

# Define function to save cluster heatmap in local


def save_heatmap(folder, img):

    if not os.path.exists(gramcam_path+folder):
        os.makedirs(gramcam_path+folder)

    if os.path.exists(gramcam_path+folder+'/'+img):
        return

    True_label = int(folder_to_nums[folder])

    img_path = imagenet_path+folder+'/'+img
    image = Image.open(img_path).convert('RGB')
    image = loader(image).float()
    image = torch.unsqueeze(image, 0)

    # Make Prediction
    y_hat = softmax(model(image.cuda()).data.cpu()[0]).numpy()
    top_5 = np.argsort(y_hat)[::-1][:5]

    cluster_use_final = clusterify_resnet50.get_cluster(top_5)

    for cluster in cluster_use_final:
        if True_label in cluster:
            IOX_cluster = BP_logP(cluster, image, 'layer4', model)

            saliency = np.clip(IOX_cluster, a_min=0, a_max=None)
            image = imsc(image[0])

            plt.imshow(image)
            plt.imshow(saliency, cmap='jet', alpha=0.4)
            plt.axis('off')
            plt.savefig(gramcam_path+folder+'/'+img)

            return


for folder in os.listdir(imagenet_path):

    for img in os.listdir(imagenet_path+folder+'/'):

        save_heatmap(folder, img)

        # pool = multiprocessing.Pool(10)
        # pool.apply_async(func=save_heatmap, args=(folder, img))

# pool.close()
# pool.join()
