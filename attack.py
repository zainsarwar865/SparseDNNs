import os
import sys
import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--root_hash_config', type=str)
parser.add_argument('--base_dir')
parser.add_argument('--mt_hash_config', type=str)
parser.add_argument('--num_eval_epochs', type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--lr', dest='lr', type=float)
parser.add_argument('--c', type=float)
parser.add_argument('--d', type=float)
parser.add_argument('--steps',  type=int)
parser.add_argument('--attack',  type=str)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--num_classes', type=int)
parser.add_argument('--original_dataset', type=str)
parser.add_argument('--trainer_type', type=str)
parser.add_argument('--gpu', type=str)
parser.add_argument('--attack_split', type=str)
parser.add_argument('--detector_type', type=str)
parser.add_argument('--total_attack_samples', type=int)
parser.add_argument('--total_train_samples', type=int)
parser.add_argument('--integrated', type=str)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--seed', type=int, help="seed for pandas sampling")

args = parser.parse_args()

args.distributed = False

if args.integrated == "True":
    args.integrated = True
else:
    args.integrated = False


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
#from torchvision.models import resnet18
from utils.resnet import resnet18
from utils import configs
import hashlib
import logging
sys.path.append("/home/zsarwar/Projects/SparseDNNs/adversarial-attacks-pytorch")
import torchattacks

from utils.utils_2 import imshow, get_pred
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.rbf_model import RBF_SVM
import time
import utils.utils as utils
from enum import Enum
import random
import torch.backends.cudnn as cudnn
from utils.wide_resnet import WideResNet

if args.integrated:
    if args.detector_type == 'Quantized':
        from torchattacks.attacks.cw_integrated_quantized import CW_RBF as CW
    elif args.detector_type == 'Regular':
        from torchattacks.attacks.cw_integrated import CW_RBF as CW
else:
    from torchattacks import CW, DeepFool

# Set seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

print("Starting attack...")

# GO-GO-GO!
normalize  =  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    
transformation = transforms.Compose([
    transforms.ToTensor(), 
    #normalize
    ])
dataset_path = configs.dataset_root_paths[args.original_dataset]

if args.attack_split == "train":
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                                download=False, transform=transformation)
    random_indices = np.random.randint(low=0, high = len(trainset), size=(args.total_attack_samples))
    trainset = Subset(trainset, indices=random_indices)
    dataloader = DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.workers)

elif args.attack_split == 'test':
    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                        download=False, transform=transformation
                                            )
    random_indices = np.random.randint(low=0, high = len(testset), size=(args.total_attack_samples))
    testset = Subset(testset, indices=random_indices)
    dataloader = DataLoader(testset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.workers)
   

loc = 'cuda:{}'.format(0)
device = 'cuda:0'


# Load best checkpoint
root_config = args.root_hash_config
root_config_hash = (hashlib.md5(root_config.encode('UTF-8')))
mt_config = root_config + "_" + root_config_hash.hexdigest()

# Every file should be created inside this directory
mt_root_directory = os.path.join(args.base_dir, mt_config)

# Check trainer type
if args.trainer_type == "MT_Baseline":
    expr_config = args.mt_hash_config 

expr_hash = (hashlib.md5(expr_config.encode('UTF-8')))
expr_name = args.trainer_type + "_" + expr_hash.hexdigest()
expr_dir = os.path.join(mt_root_directory, expr_name)

# Load checkpoint path
expr_dir = os.path.join(mt_root_directory, expr_name)
ckpt_config = "Checkpoints/model_best.pth.tar"
ckpt_path = os.path.join(expr_dir, ckpt_config)

if args.model == 'wideresnet':
    # Load model
    ###################################################################
    model = WideResNet()
    model_path = "/home/zsarwar/Projects/SparseDNNs/adversarial-attacks-pytorch/demo/models/cifar10/L2/Standard.pt"
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    ###################################################################

elif args.model == 'resnet18':
    model = resnet18()
    model.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True)
    checkpoint = torch.load(ckpt_path, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'])

model = model.to(device)

########################################################################
# Filter out incorrectly classified samples
model = model.eval()
def run_validate(loader, base_progress=0):
    all_masks = None
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):
            i = base_progress + i
            if args.gpu is not None and torch.cuda.is_available():
                images = images.cuda(device, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(device, non_blocking=True)
            # compute output
            output, relu_feats = model(images)
            # measure accuracy and record loss
            _, pred = output.topk(1, largest=True)
            pred = pred.t()
            correct = pred.eq(target[None])
            if isinstance(all_masks, torch.Tensor):
                all_masks = torch.cat((all_masks, correct), dim=1)
            else:
                all_masks = correct
    return all_masks

mask = run_validate(dataloader)
good_indices = torch.where(mask == True)[1].tolist()
# Create new subsets
if args.attack_split == "train":
    trainset = Subset(trainset, indices=good_indices)
    dataloader = DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.workers)

elif args.attack_split == 'test':

    testset = Subset(testset, indices=good_indices)
    dataloader = DataLoader(testset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.workers)
   
mask = run_validate(dataloader)
good_indices = torch.where(mask == True)[1].tolist()

########################################################################


if args.integrated:
    # Load RBF model
    rbf_config = f"RBF_{args.attack}_{args.total_train_samples}_{args.detector_type}.pkl"
    rbf_path = os.path.join(expr_dir, "RBF", rbf_config)

    with open(rbf_path,'rb') as in_model:
        clf = pickle.load(in_model)
        
    # Class RBF model
    rbf_svm = RBF_SVM(clf)
    rbf_svm = rbf_svm.to(device)

# Start attack
#mean = (0.4914, 0.4822, 0.4465)
#std = (0.247, 0.243, 0.261)

all_adv_images = None
all_og_images = None
all_labels = None

if args.integrated:
    if args.attack == "CW":
        atk = CW(model, rbf_svm, c=args.c, d=args.d,  steps=args.steps, lr=args.lr)
else:     
    if args.attack == "CW":
        atk = CW(model, c=args.c,  steps=args.steps, lr=args.lr)
    elif args.attack == "DeepFool":
        atk = DeepFool(model, steps=50, overshoot=0.02)

#atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])

for images, labels in dataloader:
    adv_images, og_images = atk(images, labels)
    if isinstance(all_adv_images, torch.Tensor):
        all_adv_images = torch.concat((all_adv_images, adv_images), dim=0)
        all_og_images = torch.concat((all_og_images, og_images), dim=0)
    else:
        all_adv_images = adv_images
        all_og_images = og_images
    if isinstance(all_labels, torch.Tensor):
        all_labels = torch.concat((all_labels, labels), dim=0)
    else:
        all_labels = labels

# Save adversarial images
image_save_config = f"Adversarial_Datasets/{args.attack}_adv_samples_{args.total_attack_samples}_{args.attack_split}_detector-type-{args.detector_type}_integrated-{args.integrated}.pickle"
image_save_path  = os.path.join(expr_dir, image_save_config)

adv_dataset = [all_adv_images, all_labels]
ben_dataset = [all_og_images, all_labels]
with open(image_save_path, 'wb') as out_dataset:
    pickle.dump(adv_dataset, out_dataset)

image_save_config = f"Benign_Datasets/{args.attack}_benign_samples_{args.total_attack_samples}_{args.attack_split}_detector-type-{args.detector_type}_integrated-{args.integrated}.pickle"
image_save_path  = os.path.join(expr_dir, image_save_config)

with open(image_save_path, 'wb') as out_dataset:
    pickle.dump(ben_dataset, out_dataset)
    

print("Attack completed...")






