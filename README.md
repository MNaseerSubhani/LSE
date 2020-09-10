# Learning from Scale-Invariant Examples for Domain Adaptation in Semantic Segmentation
M.Naseer Subhani and Mohsen Ali



### Contents
0. [Introduction](#introduction)
0. [Requirements](#requirements)
0. [Setup](#setup)
0. [Implementation](#implementation)
0. [Adapted Models](#adaptedmodels)
0. [Note](#note)



### Introduction
This repo contains implementation of paper titled ["Learning from Scale-Invariant Examples for Domain Adaptation in Semantic Segmentation"](https://arxiv.org/pdf/2007.14449.pdf)
introduced in ECCV2020  

If you use this code in your research, please cite us.
~~~~
@article{subhani2020learning,
  title={Learning from Scale-Invariant Examples for Domain Adaptation in Semantic Segmentation},
  author={Subhani, M Naseer and Ali, Mohsen},
  journal={arXiv preprint arXiv:2007.14449},
  year={2020}
}
~~~~

### Requirements:
- Ubuntu 16.04 with NVIDIA Tesla K80 GPU.
- PyTorch 1.4.0
- Python 3.5

### Setup
The root directorty supposed to be "LSE/".

a. Datasets:
 - Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset.
 - Download [Cityscapes](https://www.cityscapes-dataset.com/).
 - Download [SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/download/808/). Make sure to change class id similar to cityscapes. or download synthia labels from this [Link](https://drive.google.com/file/d/1DAetOHtEmRmY2p0swaON3T_NXhV0Xcmm/view?usp=sharing) 
 - Put all datasets to "dataset/" folder.
 
b. Pretrained Initial Source Models:
 - [GTA5_VGG16-FCN8 Init](https://drive.google.com/file/d/1OyUFtf5JHOxwYwU7vprp_GzvLDiEZ1-k/view?usp=sharing).
 - [SYNTHIA_VGG16-FCN8 Init](https://drive.google.com/file/d/1ARcOirzLeC3hWlFejzKECzAd1GNp-jnS/view?usp=sharing).
 
 - Download all pretrained models and put in "init_models/"
 

### Implementation 
a. Run pip3 install -r requirements.txt.

b. Change root directory in init.py in utils.

c. Training:
 - GTA_to_Cityscapes without Focal Loss:
   ~~~~
   python3.5 LSE.py --model VGG --source gta5 --gamma 3 --beta 0.1 --focal-loss False --batch-size 1
   ~~~~
 - GTA_to_Cityscapes with Focal Loss:
   ~~~~
   python3.5 LSE.py --model VGG --source gta5 --gamma 3 --beta 0.1 --focal-loss True --batch-size 1
   ~~~~
 - SYNTHIA_to_Cityscapes without Focal Loss:
   ~~~~
   python3.5 LSE.py --model VGG --source synthia --gamma 3 --beta 0.1 --focal-loss False --batch-size 1
   ~~~~
 - SYNTHIA_to_Cityscapes with Focal Loss:
   ~~~~
   python3.5 LSE.py --model VGG --source synthia --gamma 3 --beta 0.1 --focal-loss True --batch-size 1
   ~~~~
 
d. Evaluation:
   ~~~~
   python3.5 eval.py model-name gta5_to_Cityscape
   ~~~~
   


### Adapted Models



 - 
 ### Note
Contact: msee16021@itu.edu.pk

 
 
   
