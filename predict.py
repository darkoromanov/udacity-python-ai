#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Diego Imbriani
# DATE CREATED: 22 Jun 2018
# REVISED DATE:
##

import argparse
import torch
from torchvision import datasets, transforms, models
import cnnmodel
import numpy as np
from PIL import Image
from torch.autograd import Variable
import json

def load_model(checkpoint_path, gpu):
    print("Loading checkpoint....", end="\r")
    data = torch.load(checkpoint_path)
    cnn = cnnmodel.PreTrainedCNN(data["arch"], gpu, data["hidden_units"], pretrained=False)
    cnn.model.class_to_idx = data["class_to_idx"]
    cnn.model.load_state_dict(data['state_dict'])
    print("Loading checkpoint.... OK\n")
    return cnn

def get_input_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to image")
    parser.add_argument("checkpoint", default="checkpoint.pth", help="Path to checkpoint file")
    parser.add_argument("--top_k", default="3", help="Number of predictions", type=int)
    parser.add_argument("--gpu", default="True", help="Enable GPU", type=bool)
    parser.add_argument("--category_names", help="Categories to real names map file in JSON format")

    return parser.parse_args()

def process_image(image):
    im = Image.open(image)
    
    size = 256, 256
    im.thumbnail(size)
    
    new_size = 224
    center = 256 / 2
    coo_left = center - new_size / 2
    coo_right = center + new_size / 2
    crop = (coo_left, coo_left, coo_right, coo_right)
    
    im = im.crop(crop)
    np_image = np.array(im) / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    return torch.from_numpy(np_image.transpose(2, 0, 1)).float()

def predict(model, image_path, gpu, topk=3):
    model.eval()

    im = process_image(image_path)
    data = Variable(im) 
    input = torch.FloatTensor(data)
    
    if gpu:
        input = input.cuda()
        
    input.unsqueeze_(0)
        
    output = model.forward(input)
    ps = torch.exp(output).data
    
    top = ps.topk(topk)
    return top

in_args = get_input_args()
cnn = load_model(in_args.checkpoint, in_args.gpu)
prob, cats = predict(cnn.model, in_args.input, in_args.gpu, in_args.top_k)
with open(in_args.category_names) as f:
    names = json.load(f)
idx_to_class = {}

for m in cnn.model.class_to_idx:
    idx_to_class[cnn.model.class_to_idx[m]] = m

cats = cats.cpu().numpy()
prob = prob.cpu().numpy()
    
cat_names = [names[idx_to_class[c]] for c in cats[0]]


for i in range(0, in_args.top_k):
    print(cat_names[i] + " : " + ("%.2f" % round((prob[0][i] * 100), 2)) + "%")