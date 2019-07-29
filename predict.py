"""
Libaries
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL import Image
from torch import nn
from torchvision import datasets, models, transforms

from collections import OrderedDict
import json

import sys
from argparse import ArgumentParser

"""
Functions
"""
def load_checkpoint(path):
    checkpoint = torch.load('checkpoint.pth')
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    
    return model

def process_image(filepath):

    pil_image = Image.open(filepath)
    
    width, height = pil_image.width, pil_image.height
    
    #resize
    if width == height:
        pil_image.thumbnail((256, 256))
    
    elif width > height:
        pil_image.thumbnail((9999999,256))
    
    else: #height > width
        pil_image.thumbnail((256,9999999))
        
    #crop the center 224x224 pixels    
    crop_left = (pil_image.width - 224) / 2
    crop_right = crop_left + 224
   
    crop_top = (pil_image.height + 224) / 2
    crop_bottom = crop_top - 224
    
    cropped_image = pil_image.crop((crop_left,
                                    crop_bottom,
                                    crop_right,
                                    crop_top))
    
    #convert to np array
    numpy_image = np.array(cropped_image) / 255
    
    #scale values to be between 0 and 1
   # numpy_image = numpy_image / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    stdev = np.array([0.229, 0.224, 0.225])
    
    scaled_image = (numpy_image - mean) / stdev
    
    return np.transpose(scaled_image, (2,0,1))
    
    
def predict(filepath, model, topk):
    processed_image = process_image(filepath)
    
    #convert from numpy array to tensor
    tensor_image = torch.from_numpy(processed_image)
    
    #convert tensor type to float, add dim of length 1
    tensor_image = tensor_image.type(torch.FloatTensor).unsqueeze_(dim=0)
    
    tensor_image = tensor_image.to(device)
    
    model.to(device)
    model.eval()
    output = model.forward(tensor_image)
    
    #sigmoid
    ps = torch.exp(output)
    
    top_prob, top_class = ps.topk(topk)
    
    #bring to CPU if on CUDA
    if device == torch.device('cuda:0'):
        top_prob, top_class = top_prob.cpu(), top_class.cpu()
    
    probs, class_labels = top_prob.detach().numpy().tolist()[0], top_class.detach().numpy().tolist()[0]
    
    #map codes to labels
    idx_to_class = {val:key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[label] for label in class_labels]
    
    return probs, classes    
"""
Parse user inputs
"""
print("Parsing user input...")
parser = ArgumentParser()

parser.add_argument('--checkpoint', type = str, help = 'Saved checkpoint model filepath')
parser.add_argument('--image', type = str)
parser.add_argument('--topk', type = int)
parser.add_argument('--cat_to_name', type = str)
parser.add_argument('--use_gpu', type = str, help = 'Y/N - default to cpu if cuda not available')

args, _ = parser.parse_known_args()

checkpoint = args.checkpoint if args.checkpoint else 'checkpoint.pth'
image = args.image if args.image else 'flowers/test/100/image_07896.jpg'
topk = args.topk if args.topk else 5
cat_to_name = args.cat_to_name if args.cat_to_name else 'cat_to_name.json'

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.use_gpu:
    if args.use_gpu == 'Y':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
print("Inputs parsed.")

"""
Prediction
"""


#with open('cat_to_name.json', 'r') as f:
#    cat_to_name = json.load(f)

print("Making prediction...")
model = load_checkpoint(checkpoint)

probs, classes = predict(image, model, topk)



print("Top {} Classes: {}".format(topk, classes))
print("Top {} Probabilities: {}".format(topk, probs))
