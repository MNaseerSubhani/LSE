import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import torch.nn as nn
import numpy as np
from model import CreateModel
from torch.utils import data
import tqdm
import time
from utils import evaluation
import argparse
from utils import root_base


def parse_args():
    parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
    parser.add_argument("--model", type=str, default='VGG',help="available options : DeepLab and VGG")
    parser.add_argument("--data-dir-target", type=str, default=root_base + '/dataset/cityscapes', help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list-target", type=str, default=root_base +'/dataset/cityscapes_list/val.txt', help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.") 
    parser.add_argument("--num-classes", type=int, default=19, help="Number of classes for cityscapes.")
    parser.add_argument("--init-weights", type=str, default=None, help="initial model.")
    parser.add_argument("--set", type=str, default='val', help="choose adaptation set.")
    parser.add_argument("--model-name", type=str, default=None, help="Model's file name")
    parser.add_argument("--restore-from", type=str, default=root_base+'/snapshots/', help="Restore model's folder.") 
    parser.add_argument("--save", type=str, default=root_base+'/dataset/cityscapes/results', help="Path to save result.")    
    parser.add_argument('--gt_dir', type=str, default = root_base +'/dataset/cityscapes/gtFine/val', help='directory which stores CityScapes val gt images')
    parser.add_argument('--devkit_dir', default=root_base+'/dataset/cityscapes_list', help='base directory of cityscapes')         
    return parser.parse_args()
classes = ['road' , 'side walk', 'building' , 'wall', 'fence', 'pole', 'trafic lights', 'trafic sign', 'vegitation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck','bus', 'train', 'motorcycle', 'bicycle']


def load_model(args):
    
    args.restore_from = root_base+'/snapshots/' + args.model_name
    model, optimizer = CreateModel(args)

    return model

if __name__ == "__main__":
    args = parse_args()
    model = load_model(args)

    _,_ = evaluation.main(model)
    