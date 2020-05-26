import argparse
import os.path as osp
root_base = '/home/olektra_gpu/Desktop/olektra projects/ECCV'
restore_model = 'vggfcn_gta5_init' #'vggfcn_gta5_init'
class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
        parser.add_argument("-ft","--model", type=str, default='VGG',help="available options : DeepLab and VGG")
        parser.add_argument("-j2","--data-dir-target", type=str, default=root_base + '/dataset/cityscapes', help="Path to the directory containing the source dataset.")
        parser.add_argument("-k3","--data-list-target", type=str, default=root_base +'/dataset/cityscapes_list/val.txt', help="Path to the file listing the images in the source dataset.")
        parser.add_argument("-l4","--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.") 
        parser.add_argument("-m5","--num-classes", type=int, default=19, help="Number of classes for cityscapes.")
        parser.add_argument("-n6","--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument("-o7","--restore-from", type=str, default=root_base + '/init_models/'+restore_model, help="Where restore model parameters from.")
        parser.add_argument("-p8","--set", type=str, default='val', help="choose adaptation set.")  
        parser.add_argument("-q9","--save", type=str, default=root_base+'/dataset/cityscapes/results', help="Path to save result.")    
        parser.add_argument("-r10",'--gt_dir', type=str, default = root_base +'/dataset/cityscapes/gtFine/val', help='directory which stores CityScapes val gt images')
        parser.add_argument("-s11",'--devkit_dir', default=root_base+'/dataset/cityscapes_list', help='base directory of cityscapes')         
        return parser.parse_args()
    
