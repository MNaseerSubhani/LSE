# Learning from Scale-Invariant Examples for Domain Adaptation in Semantic Segmentation
M.Naseer Subhani and Mohsen Ali



### ToC
0. [Introduction](#introduction)
0. [Citation and license](#citation)
0. [Requirements](#requirements)
0. [Setup](#models)
0. [Usage](#usage)
0. [Results](#results)
0. [Note](#note)

### Requirements:

### Introduction

### Setup
The root directorty supposed to be "LSE/".

a. Datasets:
 - Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset.
 - Download [Cityscapes](https://www.cityscapes-dataset.com/).
 - Download [SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/download/808/). Make sure to change class id similar to cityscapes.
 - put all data to "dataset/" folder.
 
b. Pretrained Initial Source Models:
 - [GTA5_VGG16-FCN8 Init](https://drive.google.com/file/d/1OyUFtf5JHOxwYwU7vprp_GzvLDiEZ1-k/view?usp=sharing).
 - [SYNTHIA_VGG16-FCN8 Init](https://drive.google.com/file/d/1ARcOirzLeC3hWlFejzKECzAd1GNp-jnS/view?usp=sharing).
 
 - Download all pretrained model and put in "init_models/"
 




### Implementation 
a. Run pip3 install -r requirements.txt.

b. Change root directory in LSE.py.

c. Training:
 - GTA_to_Cityscapes without Focal Loss:
   ~~~~
   python3.5 LSE.py --model VGG --source gta5 --gamma 3 --beta 0.1 --focal-loss True --batch-size 1
   ~~~~
 - GTA_to_Cityscapes with Focal Loss:
 - SYNTHIA_to_Cityscapeswith Focal Loss:
 - SYNTHIA_to_Cityscapeswith with Focal Loss:
 
d. Evaluation:
 

### Results:
### Citation

 - 
 ### Note
Contact: msee16021@itu.edu.pk

 
 
   
