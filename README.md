# Learning from Scale-Invariant Examples for Domain Adaptation in Semantic Segmentation
M.Naseer Subhani and Mohsen Ali


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

 
 
   
