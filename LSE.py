import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import torch.nn as nn
import numpy as np
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from model import CreateModel
from utils.timer import Timer
import tensorboardX
from matplotlib import pyplot as plt
import random
from PIL import Image
from skimage.transform import resize
from IPython.display import clear_output
import os.path as osp
import collections
from torch.utils import data
from PIL import Image
import tqdm
import time
import argparse
from utils import root_base

IMG_W = 1024                                                           #Image width
IMG_H = 512 
def parse_args():
    parser = argparse.ArgumentParser(description="Self Supervised Adaptation")
    parser.add_argument("--model", type=str, default='VGG',help="available options : DeepLab and VGG")
    parser.add_argument("--source", type=str, default='gta5',help="source dataset : gta5 or synthia")
    parser.add_argument("--target", type=str, default='cityscapes',help="target dataset : cityscapes")
    parser.add_argument("--batch-size", type=int, default=1, help="input batch size.")
    parser.add_argument("--data-dir", type=str, default=root_base + '/dataset/', help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=root_base + '/dataset/source_list', help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data-dir-target", type=str, default=root_base + '/dataset/cityscapes', help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=root_base + '/dataset/cityscapes_list/train.txt', help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="initial learning rate for the segmentation network.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=19, help="Number of classes for cityscapes.")
    parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")
    parser.add_argument("--sorted-list", type=str, default=root_base +'/dataset/cityscapes_list/sorted.txt', help="Target data's sorted list")
    parser.add_argument("--data-gen-list", type=str, default=root_base +'/dataset/cityscapes_list/data_gen_list.txt', help="Generated Scale-Invariant patch data list")
    parser.add_argument("--generated-data", type=str, default=root_base +'/dataset/generated_data/' , help="Generated Scale-Invariant data path")
    parser.add_argument("--set", type=str, default='train', help="choose adaptation set.")
    parser.add_argument("--p", type=float, default=0.1, help="Selected portion p")
    parser.add_argument("--no-of-patches-per-image", type=int, default=4, help="Number of patches per image")
    parser.add_argument("--gamma", type=int, default=3, help="Gamma value")
    parser.add_argument("--saving-step", type=int, default=500, help="Save model steps")
    parser.add_argument("--beta", type=float, default=0.1, help="Gamma value")
    parser.add_argument("--num-steps", type=int, default= 250000, help="Number of steps")
    parser.add_argument("--focal-loss", type=bool, default=True, help="Focal Loss flag")
    parser.add_argument("--epoch-per-round", type=int, default=2, help="Epoch per round")
    parser.add_argument("--shuffel_", type=bool, default=False, help="Shuffle")
    parser.add_argument("--snapshot-dir", type=str, default=root_base + '/snapshots/', help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--num-workers", type=int, default=2, help="number of threads.")
       
    return parser.parse_args()
classes = ['road' , 'side walk', 'building' , 'wall', 'fence', 'pole', 'trafic lights', 'trafic sign', 'vegitation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck','bus', 'train', 'motorcycle', 'bicycle']
def print_args(args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    
        # save to the disk
        file_name = osp.join(args.snapshot_dir, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n') 

def channel_revert(inp):
    inp = np.rollaxis(inp,axis =-1)
    inp = np.rollaxis(inp,axis =-1)
    inp = inp.reshape(1,inp.shape[0],inp.shape[1],inp.shape[2])
    return inp

def self_entropy(pred, epsilon=1e-12):
    pred = pred[0]
    p = pred * np.log(pred+ epsilon)
    map_ = -np.sum(p, -1)
    return map_

def ent_normalization(SE,Y_pre):
    tY_pre = Y_pre[0,:,:,:]
    labs = np.argmax(tY_pre, axis=-1)
    labs1 = labs.flatten()
    se1 = SE.flatten()

    uniq = np.unique(labs1)

    for un in range(uniq.shape[0]):

        t_se = se1[labs1==uniq[un]]
        t_se1 = ((t_se - t_se.min()))/((t_se.max()-t_se.min()))

        se1[labs1==uniq[un]] = t_se1

    SE = se1.reshape(SE.shape)
    return SE

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32) # means of Target Data
def load_single_image(name):
    image = Image.open(osp.join(root_base + '/dataset/cityscapes/', "leftImg8bit/%s/%s" % ('train', name))).convert('RGB')
    # resize
    image = image.resize((IMG_W,IMG_H), Image.BICUBIC)
    rl_image = np.asarray(image, np.float32)
    rl_image_rgb = rl_image.copy()
    size = rl_image.shape
    image = rl_image[:, :, ::-1]  # change to BGR
    image -= IMG_MEAN
    image = image.transpose((2, 0, 1))
    return image.copy()

class model_init():
    def __init__(self,args):    
        _t = {'iter time' : Timer()}
        model_name = args.source + '_to_' + args.target

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)   
            os.makedirs(os.path.join(args.snapshot_dir, 'logs'))

        model, optimizer = CreateModel(args)
        self.args =args
        self.model_name =model_name
        self.model = model
        self.optimizer = optimizer
        
        
    


epsilon = 1e-12
cnt_img =0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Base_Adaptation():
    def __init__(self,model_loader):
        self.args =model_loader.args
        self.model_name =model_loader.model_name
        self.model = model_loader.model
        self.optimizer = model_loader.optimizer
        self.cnt_img = 0
        self.entropy_th_class = np.ones((1,19)) #* self.args.entropy_th
        
    def sorting(self):
        print("*********** Finding most confident samples using Class based Sorting"," ***********")
        self.class_wise_sort()
 
    def class_wise_sort(self):
        self.args.data_label_folder_target = None 
        self.args.data_dir_target = root_base +'/dataset/cityscapes'
        self.args.data_list_target = root_base +'/dataset/cityscapes_list/train.txt'
        self.args.num_steps = self.args.total_no_of_target
        self.args.batch_size = 1
        self.model.eval()
        self.model.cuda()    
        targetloader = CreateTrgDataLoader(self.args)
        
        data_save_class = {classes[0]:[],classes[1]:[],classes[2]:[],classes[3]:[],classes[4]:[],classes[5]:[],classes[6]:[],classes[7]:[],classes[8]:[],classes[9]:[],classes[10]:[],classes[11]:[],classes[12]:[],classes[13]:[],classes[14]:[],classes[15]:[],classes[16]:[],classes[17]:[],classes[18]:[],"f_name_"+classes[0]:[],"f_name_"+classes[1]:[],"f_name_"+classes[2]:[],"f_name_"+classes[3]:[],"f_name_"+classes[4]:[],"f_name_"+classes[5]:[],"f_name_"+classes[6]:[],"f_name_"+classes[7]:[],"f_name_"+classes[8]:[],"f_name_"+classes[9]:[],"f_name_"+classes[10]:[],"f_name_"+classes[11]:[],"f_name_"+classes[12]:[],"f_name_"+classes[13]:[],"f_name_"+classes[14]:[],"f_name_"+classes[15]:[],"f_name_"+classes[16]:[],"f_name_"+classes[17]:[],"f_name_"+classes[18]:[]}
        data_save_class_sorted = {classes[0]:[],classes[1]:[],classes[2]:[],classes[3]:[],classes[4]:[],classes[5]:[],classes[6]:[],classes[7]:[],classes[8]:[],classes[9]:[],classes[10]:[],classes[11]:[],classes[12]:[],classes[13]:[],classes[14]:[],classes[15]:[],classes[16]:[],classes[17]:[],classes[18]:[],"f_name_"+classes[0]:[],"f_name_"+classes[1]:[],"f_name_"+classes[2]:[],"f_name_"+classes[3]:[],"f_name_"+classes[4]:[],"f_name_"+classes[5]:[],"f_name_"+classes[6]:[],"f_name_"+classes[7]:[],"f_name_"+classes[8]:[],"f_name_"+classes[9]:[],"f_name_"+classes[10]:[],"f_name_"+classes[11]:[],"f_name_"+classes[12]:[],"f_name_"+classes[13]:[],"f_name_"+classes[14]:[],"f_name_"+classes[15]:[],"f_name_"+classes[16]:[],"f_name_"+classes[17]:[],"f_name_"+classes[18]:[]}
    
        for index, batch in tqdm.tqdm(enumerate(targetloader)):
            
            image, _, name ,__= batch
            fn = name[0]
            output = self.model(Variable(image).cuda())
            output = nn.functional.softmax(output, dim=1)
            output=output.cpu().data[0].numpy()
            y_pred = channel_revert(output)
            MP_max = np.max(y_pred,axis = -1)[0]
            LP_argmax = np.argmax(y_pred,axis = -1)[0]
            for c in range(len(classes) ):
                M_c = MP_max[LP_argmax == c]
                if(M_c.any()):
                    data_save_class['f_name_'+classes[c]].append(fn)
                    data_save_class[classes[c]].append(np.mean(M_c))
        
        f = open(self.args.sorted_list,"w+") 
        self.cnt_img=0
        over_all_list = []
        for c in range(len(classes) ):      
            
            if(self.args.source == 'synthia'):
                if(c !=9 and c !=14 and c !=16):
                    max_prob_sorted_ascending, f_name_arrays = zip(*sorted(zip(data_save_class[classes[c]], data_save_class['f_name_'+classes[c]])))
                    f_name_arrays = f_name_arrays[::-1]
                    max_prob_sorted_ascending = max_prob_sorted_ascending[::-1]
                    data_save_class_sorted[classes[c]] = max_prob_sorted_ascending
                    data_save_class_sorted['f_name_'+classes[c]] = f_name_arrays
                    select_imgs =  int(len(f_name_arrays) * (self.args.p/19))
                    selct = f_name_arrays[0:select_imgs]

                    ######Dynamic threshold of each class##############

                    f_n = f_name_arrays[select_imgs]
                    img= load_single_image(f_n)
                    img = img.reshape(1,3,IMG_H,IMG_W)
                    img = torch.from_numpy(img).float().to(device)
                    out_ = self.model(Variable(img).cuda())
                    out_ = nn.functional.softmax(out_, dim=1)
                    out_=out_.cpu().data[0].numpy()
                    y_pred = channel_revert(out_)

                    Y_argmax = np.argmax(y_pred,axis = -1)[0]
                    SE_main = self_entropy(y_pred) 
                    SE_main = ent_normalization(SE_main,y_pred)
                    class_se = SE_main[Y_argmax == c]
                    mean_class_se = np.mean(class_se) 
                    self.entropy_th_class[0][c] = mean_class_se

                    ##############################################

                    over_all_list += selct
            else:
                max_prob_sorted_ascending, f_name_arrays = zip(*sorted(zip(data_save_class[classes[c]], data_save_class['f_name_'+classes[c]])))
                f_name_arrays = f_name_arrays[::-1]
                max_prob_sorted_ascending = max_prob_sorted_ascending[::-1]
                data_save_class_sorted[classes[c]] = max_prob_sorted_ascending
                data_save_class_sorted['f_name_'+classes[c]] = f_name_arrays
                select_imgs =  int(len(f_name_arrays) * (self.args.p/19))
                selct = f_name_arrays[0:select_imgs]

                ######Dynamic threshold of each class##############

                f_n = f_name_arrays[select_imgs]
                img= load_single_image(f_n)
                img = img.reshape(1,3,IMG_H,IMG_W)
                img = torch.from_numpy(img).float().to(device)
                out_ = self.model(Variable(img).cuda())
                out_ = nn.functional.softmax(out_, dim=1)
                out_=out_.cpu().data[0].numpy()
                y_pred = channel_revert(out_)
                Y_argmax = np.argmax(y_pred,axis = -1)[0]
                SE_main = self_entropy(y_pred)
                SE_main = ent_normalization(SE_main,y_pred)
                class_se = SE_main[Y_argmax == c]
                mean_class_se = np.mean(class_se) 
                self.entropy_th_class[0][c] = mean_class_se
                ##############################################
                over_all_list += selct
            
        over_all_list = list(dict.fromkeys(over_all_list))
        for loop in over_all_list:
            f.write(loop+'\n')
            self.cnt_img +=1
    

    def save_data(self,patch_x, patch_y, path,i,map_):
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path+'/images')
            os.makedirs(path+'/labels')

        Y = np.argmax(patch_y,axis = -1)
        
        Y[map_ == 0] = 255
        plt.imsave( path +"images/"+ str(i)+'.png',patch_x)  #p_small_data
        Y=np.asarray(Y,dtype=np.uint8) 
        Y=Image.fromarray(Y,mode = 'L')
        Y.save(path+ "labels/"+str(i)+'.png')
        
    def Gen_Scale_Inv_Exp(self):  
        self.Generate()
            
    def Generate(self):
        self.args.data_label_folder_target = None 
        stop_point = self.cnt_img
        self.args.data_dir_target = root_base +'/dataset/cityscapes'
        self.args.data_list_target = self.args.sorted_list
        self.args.batch_size = 1
        self.model.eval()
        self.model.cuda()    
        targetloader = CreateTrgDataLoader(self.args)
        
        count = 0
        
        f = open(self.args.data_gen_list,"w+")
        print("***********Generating Random Scale Invariant Example***********")
        print("Total it : ",stop_point+1)
        for index, batch in tqdm.tqdm(enumerate(targetloader)):
            
            if(index > stop_point):
                break
            
            image, _, name, rl_img = batch
            fn = name[0]
            output = self.model(Variable(image).cuda())
            output = nn.functional.softmax(output, dim=1)
            rl_img = rl_img.cpu().data[0].numpy()/255.0
            output=output.cpu().data[0].numpy()
            Y_pred = channel_revert(output)
            
            ###############################################
            map_ = np.zeros((IMG_H,IMG_W))
            SE = self_entropy(Y_pred)
            SE = ent_normalization(SE,Y_pred)
            
            for il in range(IMG_H):
                for ij in range(IMG_W):
                    
                    if(SE[il][ij] >= self.entropy_th_class[0][np.argmax(Y_pred[0][il][ij],axis = -1)]):
                        map_[il][ij] =  0
                    else:
                        map_[il][ij] =  1
     
            for patch in range(self.args.no_of_patches_per_image):
                
                if(self.args.patch_size[1] == 1024):
                    X_Patch_resize = rl_img
                    Y_Patch_resize = Y_pred
                    map_patch_re = map_
                    
                else:

                    point_X_s = self.args.patch_size[1]
                    point_Y_s = self.args.patch_size[0]
                    point_X = random.randint(0, IMG_W-point_X_s)
                    point_Y = random.randint(0, IMG_H - point_Y_s)
                    x_PATCH =  rl_img[ point_Y:point_Y+point_Y_s, point_X:point_X+point_X_s].reshape(point_Y_s,point_X_s,3)
                    y_PATCH =  Y_pred[0, point_Y:point_Y+point_Y_s, point_X:point_X+point_X_s].reshape(1,point_Y_s,point_X_s,19)
                    map_patch = map_[point_Y:point_Y+point_Y_s, point_X:point_X+point_X_s].reshape(point_Y_s,point_X_s)

                    ################resize  ###############
                    x_PATCH = np.rollaxis(x_PATCH, axis = -1).reshape(1,3,point_Y_s,point_X_s)
                    x_PATCH = torch.from_numpy(x_PATCH).float().to(device) 
                    y_PATCH = np.rollaxis(y_PATCH[0], axis = -1).reshape(1,19,point_Y_s,point_X_s)
                    y_PATCH = torch.from_numpy(y_PATCH).float().to(device)                    
                    map_patch = map_patch.reshape(1,1,point_Y_s,point_X_s)
                    map_patch = torch.from_numpy(map_patch).float().to(device)
                    
                    X_Patch_resize = nn.functional.upsample(x_PATCH, (IMG_H, IMG_W), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                    X_Patch_resize = np.rollaxis(X_Patch_resize, axis = -1)
                    X_Patch_resize = np.rollaxis(X_Patch_resize, axis = -1)
                    
                    
                    Y_Patch_resize = nn.functional.upsample(y_PATCH, (IMG_H, IMG_W), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                    Y_Patch_resize = np.rollaxis(Y_Patch_resize, axis = -1)
                    Y_Patch_resize = np.rollaxis(Y_Patch_resize, axis = -1).reshape(1,IMG_H,IMG_W,19)
                    
                    
                    map_patch_re = nn.functional.upsample(map_patch, (IMG_H, IMG_W), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                    map_patch_re = map_patch_re[0]
                         
                f.write(str(count+patch)+".png"+'\n')
                self.save_data(X_Patch_resize, Y_Patch_resize.reshape(IMG_H,IMG_W,19), self.args.generated_data,(count+patch),map_patch_re)

            count+=self.args.no_of_patches_per_image + 1
        print("*****************END****************")
            
            
    def train_adp(self,round_):
        self.args.data_label_folder_target = root_base +'/dataset/generated_data/'   
        self.args.shuffel_ = True
        self.args.data_dir_target = root_base +'/dataset/generated_data/'
        self.args.data_list_target = self.args.data_gen_list
        self.args.batch_size = 1
 
        _t = {'iter time' : Timer()}

        self.args.num_steps = int(self.cnt_img * self.args.epoch_per_round * self.args.no_of_patches_per_image)
        
        sourceloader, targetloader = CreateSrcDataLoader(self.args), CreateTrgDataLoader(self.args)
        targetloader_iter, sourceloader_iter = iter(targetloader), iter(sourceloader)
        start_iter = 0

        train_writer = tensorboardX.SummaryWriter(os.path.join(self.args.snapshot_dir, "logs", self.model_name))

        cudnn.enabled = True
        cudnn.benchmark = True
        self.model.train()
        self.model.cuda()

        loss = ['loss_seg_src', 'loss_seg_trg']
        _t['iter time'].tic()

        for i in range(start_iter, self.args.num_steps):

            self.model.adjust_learning_rate(self.args, self.optimizer, i)
            self.optimizer.zero_grad()
            src_img, src_lbl, _, _,_ = sourceloader_iter.next()
            src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda()
            src_seg_score = self.model(src_img, lbl=src_lbl)       
            loss_seg_src = self.model.loss
            loss_src = torch.mean(loss_seg_src)     
            ##############################
            loss_src.backward()

            trg_img, trg_lbl, _, _ = targetloader_iter.next()
            trg_img, trg_lbl = Variable(trg_img).cuda(), Variable(trg_lbl.long()).cuda()
            trg_seg_score = self.model(trg_img, lbl=trg_lbl) 
            ############################
            loss_seg_trg = self.model.loss 
            
            ##########Focal loss############
            loss_trg_2 = torch.mean(loss_seg_trg)
            if self.args.focal_loss:
                pt = torch.exp(-loss_seg_trg)
                loss_trg =   loss_seg_trg  * (1-pt)**self.args.gamma
                trg_fcl = torch.mean(loss_trg)
            else:
                trg_fcl = 0
        
            loss_trg =  self.args.beta *trg_fcl  + loss_trg_2
            loss_trg.backward()
        
            src_seg_score, trg_seg_score = src_seg_score.detach(), trg_seg_score.detach()
            self.optimizer.step()

            if (i+1) % self.args.saving_step == 0 :
                print ('taking snapshot ...')
                if(args.focal_loss):
                    torch.save(self.model.state_dict(), os.path.join(self.args.snapshot_dir, '%s' %(self.args.source+"_to_" +self.args.target)+"_w_focal_loss_"+args.model +'.pth' ))   
                else:
                    torch.save(self.model.state_dict(), os.path.join(self.args.snapshot_dir, '%s' %(self.args.source+"_to_" +self.args.target)+"_wo_focal_loss" +args.model +'.pth' ))   
            if (i+1) % 100 == 0:
                _t['iter time'].toc(average=False)
                print ('[it %d][src seg loss %.4f][lr %.4f][%.2fs]' % \
                        (i + 1, loss_src.data, self.optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff))
                
                _t['iter time'].tic()

       

if __name__ == "__main__":
    args = parse_args()

    if args.model == 'VGG' and args.source == 'gta5':
        model_weights = 'gta5_vggfcn_init.pth'
    elif args.model == 'VGG' and args.source == 'synthia':
        model_weights = 'synthia_vggfcn_init.pth'
    elif args.model == 'DeepLab' and args.source == 'gta5':
        model_weights = 'gta5_deeplab_resnet101_init'
    elif args.model == 'DeepLab' and args.source == 'synthia':
        model_weights = 'synthia_deeplab_resnet101_init'

    if args.model == 'DeepLab':
        args.restore_from   = root_base + '/init_models/' + model_weights   
    else:
        args.init_weights   = root_base + '/init_models/' + model_weights 
        args.restore_from = None  
    
    args.data_dir = root_base + '/dataset/'+args.source
    args.data_list = root_base+'/dataset/'+args.source+'_list/train.txt'
    args.data_list_target = root_base +'/dataset/cityscapes_list/train.txt'
    

    model_initl = model_init(args)

    Rounds = int(1/model_initl.args.p)
    model_initl.args.total_no_of_target = 2975         # Total number of target images
    model_initl.args.patch_size = (256,512)

    print_args(args)
    Base = Base_Adaptation(model_initl)
  
    
    for round_ in range(Rounds):
        print("############### Round ",round_," #####################")
        Base.sorting()
        Base.Gen_Scale_Inv_Exp()
        Base.train_adp(round_)
        Base.args.p = Base.args.p+0.05
 
        if(Base.args.p > 0.4):
            break

      
        

    