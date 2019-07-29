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
def validate(model, criterion, dataloader):
    loss = 0
    accuracy = 0
    
    for images, labels in iter(dataloader):
        
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return loss / len(dataloader), accuracy / len(dataloader)


"""
Parse user inputs
"""
print("Parsing user inputs...")
parser = ArgumentParser()

parser.add_argument('--data_dir', type = str)
parser.add_argument('--checkpoint', type = str)
parser.add_argument('--learning_rate', type = float)
parser.add_argument('--epochs', type = int)
parser.add_argument('--use_gpu', type = str, help = 'Y/N - default to cpu if cuda not available')
parser.add_argument('--arch', type = str, help = 'NN architecture - choose VGG13 or VGG16')
parser.add_argument('--hidden_layer1', type = int, help = 'Number of units in first layer (NN must have two hidden layers)')
parser.add_argument('--hidden_layer2', type = int, help = 'Number of units in second layer (NN must have two hidden layers)')

args, _ = parser.parse_known_args()

data_dir = args.data_dir if args.data_dir else 'flowers'
checkpoint = args.checkpoint if args.checkpoint else 'checkpoint.pth'

learning_rate = args.learning_rate if args.learning_rate else 0.001
epochs = args.epochs if args.epochs else 5

hidden_layer1 = args.hidden_layer1 if args.hidden_layer1 else 4096
hidden_layer2 = args.hidden_layer2 if args.hidden_layer2 else 2048

arch = args.arch if args.arch else 'VGG16'

if args.use_gpu:
    if args.use_gpu == 'Y':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Parsing Complete")

"""
Set up data loading
"""
print("Setting up data loaders...")
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
                                          

validate_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
validate_data = datasets.ImageFolder(valid_dir, transform = validate_transforms)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True) 
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
validate_dataloader = torch.utils.data.DataLoader(validate_data, batch_size = 64, shuffle = True) 
print("Data loader setup complete.")

"""
Construct model
"""
print("Constructing network...")

if arch == 'VGG16':
    model = models.vgg16(pretrained = True)
else:
    model = models.vgg13(pretrained = True)
    
for param in model.parameters():
    param.requires_grad = False

input_size = 25088
output_size = 102

classifier = nn.Sequential(
    nn.Linear(input_size, hidden_layer1),
    nn.ReLU(),
    nn.Linear(hidden_layer1, hidden_layer2),
    nn.ReLU(),
    nn.Linear(hidden_layer2, output_size),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier

#device-agnostic CUDA code:
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr = learning_rate)

print_every = 40
steps = 0

model.to(device)

running_loss = 0

print ("Training model...")

for e in range(epochs):
    model.train()
    
    for images, labels in iter(train_dataloader):
        steps += 1
        
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            
            with torch.no_grad():
                validate_loss, validate_accuracy = validate(model, criterion, validate_dataloader)
                
            print("Epoch: {}/{} | ".format(e+1, epochs),
                  "Loss: {:.4f} | ".format(running_loss/print_every),
                  "Validation Loss: {:.4f} | ".format(validate_loss),
                  "Validation Accuracy: {:.4f}".format(validate_accuracy)
                                                     )
            running_loss = 0
        
            model.train()

print("Training Complete.")

model.class_to_idx = train_data.class_to_idx
checkpoint_params = {
    'epochs': epochs,
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'class_to_idx': model.class_to_idx,
    'arch': arch,
    'learning rate': learning_rate,
}

torch.save(checkpoint_params,checkpoint)