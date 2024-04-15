import os
import sys
import argparse
import signal
import sys

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--root_hash_config', type=str)
parser.add_argument('--base_dir')
parser.add_argument('--mt_hash_config', type=str)
parser.add_argument('--num_eval_epochs', type=int)
parser.add_argument('--workers', default=2, type=int)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--lr', dest='lr', type=float)
parser.add_argument('--c', type=float)
parser.add_argument('--d', type=float)
parser.add_argument('--c_attack', type=float)
parser.add_argument('--d_attack', type=float)
parser.add_argument('--eps', type=float)
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
parser.add_argument('--scale_factor', type=int),
parser.add_argument('--sparsefilter', type=str),
parser.add_argument('--seed', type=int, help="seed for pandas sampling")

args = parser.parse_args()

def timeout_handler(signum, frame):
    print("Script completed after x seconds.")
    sys.exit(0)

x_seconds = 14000

# Set the signal handler
signal.signal(signal.SIGALRM, timeout_handler)

# Set the alarm to trigger after x_seconds
signal.alarm(x_seconds)

def run_validate(model, loader, base_progress=0):
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
            #output, relu_feats = model(images)
            output = model(images)
            # measure accuracy and record loss
            _, pred = output.topk(1, largest=True)
            pred = pred.t()
            correct = pred.eq(target[None])
            if isinstance(all_masks, torch.Tensor):
                all_masks = torch.cat((all_masks, correct), dim=1)
            else:
                all_masks = correct
    return all_masks


args.distributed = False

if args.integrated == "True":
    args.integrated = True
else:
    args.integrated = False


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.gpu = 0

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
#from torchvision.models import resnet18

from utils.resnet_rand import SparsifyFiltersLayer, SparsifyKernelGroups
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

import time
import utils.utils as utils
from enum import Enum
import random
import torch.backends.cudnn as cudnn
from utils.wide_resnet import WideResNet
import yaml
from typing import Any, Union, Type
from utils.MLP import MLP

if args.integrated:
    if args.detector_type == 'Quantized':
        from torchattacks.attacks.cw_integrated_quantized import CW_MLP as CW
    elif args.detector_type == 'Regular':
        from torchattacks.attacks.cw_integrated import CW_MLP as CW
else:
    from torchattacks.attacks.cw_bounded import CW


# Select sparseblock here
sparseblock : Type[Union[SparsifyFiltersLayer, SparsifyKernelGroups]]

if args.sparsefilter == 'SparsifyFiltersLayer':
    sparseblock = SparsifyFiltersLayer
    from utils.resnet_rand import resnet18
elif args.sparsefilter == 'SparsifyKernelGroups':
    from utils.resnet_rand import resnet18
    sparseblock = SparsifyKernelGroups
    from torchattacks.attacks.cw_rand_bounded import CW
elif args.sparsefilter == "None":
    print("Testing regular ResNEt")
    from utils.resnet import resnet18




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
    ])
dataset_path = configs.dataset_root_paths[args.original_dataset]

if args.attack_split == "train":
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                                download=False, transform=transformation)
    random_indices = np.random.randint(low=0, high = len(trainset), size=(args.total_attack_samples))
    trainset = Subset(trainset, indices=random_indices)
    dataloader = DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.workers)

elif args.attack_split == 'test':
    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                        download=False, transform=transformation
                                            )
    random_indices = np.random.randint(low=0, high = len(testset), size=(args.total_attack_samples))
    testset = Subset(testset, indices=random_indices)
    dataloader = DataLoader(testset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.workers)
   

device = 'cuda:0'
loc = device

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
mlp_ckpt_dir = os.path.join(expr_dir, "MLP")

# Create and save YAML file
expr_config_dict = {}
all_args = args._get_kwargs()
expr_config_dict = {tup[0]:tup[1] for tup in all_args}
yaml_file = os.path.join(expr_dir, f"Config_integrated_{args.integrated}.yaml")
with open(yaml_file, 'w') as yaml_out:
    yaml.dump(expr_config_dict, yaml_out)


# Load checkpoint path
expr_dir = os.path.join(mt_root_directory, expr_name)
ckpt_config = "Checkpoints/model_best.pth.tar"
ckpt_path = os.path.join(expr_dir, ckpt_config)
print("ckpt_path", ckpt_path)

if args.model == 'wideresnet':
    model = WideResNet()

elif args.model == 'resnet18':
    if args.sparsefilter == "None":
        model = resnet18()
    else:
        model = resnet18(sparsefilter=sparseblock,scale_factor=args.scale_factor)
        print("Loading randCNN...")
    
    model.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True)

elif args.model == 'MLP':
    model = MLP()


checkpoint = torch.load(ckpt_path, map_location=loc)
model.load_state_dict(checkpoint['state_dict'])

model = model.to(device)

########################################################################
# Filter out incorrectly classified samples
model = model.eval()

mask = run_validate(model,dataloader)
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
   
########################################################################

print(f"Size of dataset: {len(dataloader.dataset)} ")

"""
# Load trained MLP
if args.integrated:        
    mlp = MLP()
    best_ckpt_name = "model_best.pth.tar"
    best_ckpt_path = os.path.join(mlp_ckpt_dir, best_ckpt_name)
    if args.gpu is None:
        checkpoint = torch.load(best_ckpt_path, map_location=loc)
    elif torch.cuda.is_available():
        # Map model to be loaded to specified single gpu.
        loc = device
        # Load best checkpoint
        checkpoint = torch.load(best_ckpt_path, map_location=loc)
    best_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    best_acc1 = torch.tensor(best_acc1)
    mlp.load_state_dict(checkpoint['state_dict'])
    mlp = mlp.to(device=device)
"""
all_adv_images = None
all_og_images = None
all_labels = None
all_flipped_indices = None
all_best_moving_avg = None


if args.integrated:
    if args.attack == "CW":
        atk = CW(model, mlp, c=args.c_attack, d=args.d_attack, steps=args.steps, lr=args.lr)
else:     
    if args.attack == "CW":
        atk = CW(model, c=args.c_attack, steps=args.steps, lr=args.lr, eps=args.eps)

#atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])

for bn, (images, labels) in enumerate(dataloader):
    adv_images, og_images, flipped_indices, best_moving_avg  = atk(images, labels)
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
    if isinstance(all_flipped_indices, torch.Tensor):
        flipped_indices = flipped_indices + (args.batch_size * bn)
        all_flipped_indices = torch.concat((all_flipped_indices, flipped_indices), dim=0)
    else:
        all_flipped_indices = flipped_indices
    
    if isinstance(all_best_moving_avg, torch.Tensor):
        all_best_moving_avg = torch.concat((all_best_moving_avg, best_moving_avg), dim=0)
    else:
        all_best_moving_avg = best_moving_avg
    

# Save adversarial images
image_save_config = f"Adversarial_Datasets/{args.attack}_adv_samples_{args.total_attack_samples}_{args.attack_split}_detector-type-{args.detector_type}_integrated-{args.integrated}_c-{args.c_attack}_d-{args.d_attack}_eps-{args.eps}.pickle"
image_save_path  = os.path.join(expr_dir, image_save_config)

adv_dataset = [all_adv_images, all_labels]
ben_dataset = [all_og_images, all_labels]
with open(image_save_path, 'wb') as out_dataset:
    pickle.dump(adv_dataset, out_dataset)

image_save_config = f"Benign_Datasets/{args.attack}_benign_samples_{args.total_attack_samples}_{args.attack_split}_detector-type-{args.detector_type}_integrated-{args.integrated}_c-{args.c_attack}_d-{args.d_attack}_eps-{args.eps}.pickle"
image_save_path  = os.path.join(expr_dir, image_save_config)

with open(image_save_path, 'wb') as out_dataset:
    pickle.dump(ben_dataset, out_dataset)

perturbed_indices_save_config = f"Predictions/Perturbed_Samples/{args.attack}_benign_samples_{args.total_attack_samples}_{args.attack_split}_detector-type-{args.detector_type}_integrated-{args.integrated}_c-{args.c_attack}_d-{args.d_attack}_eps-{args.eps}.pt"
perturbed_indices_save_path  = os.path.join(expr_dir, perturbed_indices_save_config)

torch.save(all_flipped_indices, perturbed_indices_save_path)


best_moving_avg_save_config = f"Predictions/Perturbed_Samples/Moving_avg_{args.attack}_benign_samples_{args.total_attack_samples}_{args.attack_split}_detector-type-{args.detector_type}_integrated-{args.integrated}_c-{args.c_attack}_d-{args.d_attack}_eps-{args.eps}.pt"
best_moving_avg_save_path  = os.path.join(expr_dir, best_moving_avg_save_config)

torch.save(all_best_moving_avg, best_moving_avg_save_path)


total_samples = len(dataloader.dataset)
total_flipped = all_flipped_indices.shape[0]
asr = total_flipped / total_samples
print("Attack stats...")
print(f"Total samples : {total_samples}")
print(f"Total flipped : {total_flipped}")
print(f"ASR : {asr}")
print("Attack completed...")
