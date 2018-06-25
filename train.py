#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Diego Imbriani
# DATE CREATED: 22 Jun 2018
# REVISED DATE:
##

import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
from collections import OrderedDict
import cnnmodel

def log(text, color="white"):
    code = "\033[1;37;40m"
    if(color == "red"):
        code = "\033[0;31;47m"
    if(color == "cyan"):
        code = "\033[1;36;40m"

    print(code + text + "\033[1;37;40m")


def get_input_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Data directory")
    parser.add_argument("--save_dir", default=".", help="Path to the directory to save checkpoints")
    parser.add_argument("--arch", default="vgg13", help="CNN model architecture to use for image classification (default 'vgg13')")
    parser.add_argument("--learning_rate", default="0.01", help="Learning rate", type=float)
    parser.add_argument("--hidden_units", help="Hidden units", default="2000", type=int)
    parser.add_argument("--epochs", help="Number of epochs for training", default="3", type=int)
    parser.add_argument("--gpu", default="True", help="Enable GPU", type=bool)

    return parser.parse_args()

def prepare_datasets(data_dir):
    log("Reading data directory", "cyan")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = { "train":  transforms.Compose([
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
                        "valid": transforms.Compose([
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
                    }

    image_datasets = { "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
                       "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"])
                     }


    dataloaders = { "train" : torch.utils.data.DataLoader(image_datasets["train"], batch_size=32, shuffle=True),
                    "valid" : torch.utils.data.DataLoader(image_datasets["valid"], batch_size=32, shuffle=True)
                  }

    return image_datasets, dataloaders
                    

def main():
    in_arg = get_input_args()
    data_sets, data_loaders = prepare_datasets(in_arg.data_dir)

    cnn = cnnmodel.PreTrainedCNN(in_arg.arch, in_arg.gpu, in_arg.hidden_units, pretrained=True)
    cnn.train(data_loaders["train"], in_arg.learning_rate, in_arg.epochs)
    cnn.validate(data_loaders["valid"])
    cnn.save(in_arg.save_dir, data_sets['train'].class_to_idx)    

if __name__ == "__main__":
    main()
