import argparse
import os.path as osp
root_base = '/home/olektra_gpu/Desktop/olektra projects/ECCV'
class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
        parser.add_argument('-fq',"--model", type=str, default='DeepLab',help="available options : DeepLab and VGG")
        parser.add_argument('-g',"--source", type=str, default='gta5',help="source dataset : gta5 or synthia")
        parser.add_argument('-d',"--target", type=str, default='cityscapes',help="target dataset : cityscapes")
        parser.add_argument('-j',"--batch-size", type=int, default=1, help="input batch size.")
        parser.add_argument('-po',"--num-workers", type=int, default=2, help="number of threads.")
        parser.add_argument('-k',"--data-dir", type=str, default=root_base + '/dataset/', help="Path to the directory containing the source dataset.")
        parser.add_argument('-a',"--data-list", type=str, default=root_base + '/dataset/source_list', help="Path to the file listing the images in the source dataset.")
        parser.add_argument('-bn',"--data-dir-target", type=str, default=root_base + '/dataset/cityscapes', help="Path to the directory containing the target dataset.")
        parser.add_argument('-bu',"--data-list-target", type=str, default=root_base + '/dataset/cityscapes_list/train.txt', help="Path to the file listing the images in the target dataset.")
        parser.add_argument('-ij',"--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.")    
        parser.add_argument('-ni',"--learning-rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
        parser.add_argument('-nj',"--learning-rate-D", type=float, default=1e-4, help="initial learning rate for discriminator.")
        parser.add_argument('-n5',"--lambda-adv-target", type=float, default=0.001, help="lambda_adv for adversarial training.")
        parser.add_argument('-nq',"--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument('-nw',"--num-classes", type=int, default=19, help="Number of classes for cityscapes.")
        parser.add_argument('-np',"--num-steps", type=int, default=250000, help="Number of training steps.")
        parser.add_argument('-n4',"--num-steps-stop", type=int, default=120000, help="Number of training steps for early stopping.")
        parser.add_argument('-ng',"--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")
        parser.add_argument('-nl',"--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument('-oi',"--restore-from", type=str, default=None, help="Where restore model parameters from.")
        parser.add_argument('-okj',"--save-pred-every", type=int, default=10000, help="Save summaries and checkpoint every often.")
        parser.add_argument('-jhg',"--print-freq", type=int, default=100, help="print loss and time fequency.")
        parser.add_argument('-pkt',"--snapshot-dir", type=str, default=root_base + '/snapshots/', help="Where to save snapshots of the model.")
        parser.add_argument('-lmk',"--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
        parser.add_argument('-njn',"--set", type=str, default='train', help="choose adaptation set.")  
        parser.add_argument('-tjn',"--shuffel_", type=bool, default=False, help="data genrater flag.")

        return parser.parse_args()
    
    def print_options(self, args):
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
        
