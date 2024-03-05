import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--root_hash_config', type=str)
parser.add_argument('--base_dir')
parser.add_argument('--mt_hash_config', type=str)
parser.add_argument('--arch', metavar='ARCH',)
parser.add_argument('--num_eval_epochs', type=int)
parser.add_argument('--workers', default=2, type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--lr', dest='lr', type=float)
parser.add_argument('--lr_warmup_epochs', type=int)
parser.add_argument('--lr_warmup_decay',  type=float)
parser.add_argument('--lr_min', default=0.0, type=float)
parser.add_argument('--label_smoothing', type=float)
parser.add_argument("--mixup_alpha",  type=float)
parser.add_argument("--cutmix_alpha", type=float)
parser.add_argument("--auto_augment_policy", default='ta_wide', type=str)
parser.add_argument("--random_erasing", type=float)
parser.add_argument("--use_v2", default=False, type=bool)
parser.add_argument("--model_ema", type=bool)
parser.add_argument("--model_ema_steps",type=int,default=32)
parser.add_argument("--model_ema_decay",type=float, default=0.99998)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--weight_decay',  type=float)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--resume', type=str)
parser.add_argument('--evaluate', dest='evaluate', default=False)
parser.add_argument('--pretrained',type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--multiprocessing_distributed',type=bool, default=False)
parser.add_argument('--seed', type=int, help="seed for pandas sampling")
parser.add_argument('--freeze_layers', type=str)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--eval_pretrained', type=str)
parser.add_argument('--new_classifier', type=str)
parser.add_argument('--test_per_class', type=str)
parser.add_argument('--trainer_type', type=str)
parser.add_argument('--original_dataset', type=str)
parser.add_argument('--original_config', type=str)
parser.add_argument('--c', type=float)
parser.add_argument('--d', type=float)
parser.add_argument('--weight_repulsion', type=str)
parser.add_argument('--scale_factor', type=int),
parser.add_argument('--sparsefilter', type=str),
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
args.gpu = 0


import random
import shutil
import time
import warnings
from enum import Enum
import yaml
import pickle
import numpy as np
import pandas as pd
from glob import glob
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as distf
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision
import torchvision.models as models
#from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Subset
from utils.dataset import CustomImageDataset
import hashlib
from sklearn.metrics import f1_score
import logging
from collections import Counter
from utils.transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
import utils.utils as utils
import utils.configs as configs
from utils.resnet_rand import resnet18
from typing import Type, Union, Any
from utils.resnet_rand import SparsifyFiltersLayer, SparsifyKernelGroups
import signal
import sys

from utils.MLP import MLP

def timeout_handler(signum, frame):
    print("Script completed after x seconds.")
    sys.exit(0)


x_seconds = 14000


# Set the signal handler
signal.signal(signal.SIGALRM, timeout_handler)

# Set the alarm to trigger after x_seconds
signal.alarm(x_seconds)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# Handle bash boolean variables
if args.pretrained == "True":
    args.pretrained = True
else:
    args.pretrained = False

if args.freeze_layers == "True":
    args.freeze_layers = True
else:
    args.freeze_layers = False



if args.new_classifier == "True":
    args.new_classifier = True
else:
    args.new_classifier = False

if args.test_per_class == "True":
    args.test_per_class = True
else:
    args.test_per_class = False


if args.eval_pretrained == "True":
    args.eval_pretrained = True
else:
    args.eval_pretrained = False



if args.model_ema == "True":
    args.model_ema = True
else:
    args.model_ema = False


if args.weight_repulsion == "True":
    args.weight_repulsion = True
else:
    args.weight_repulsion = False


if args.resume == "True":
    args.resume = True
else:
    args.resume = False



# assertions to avoid divide by 0 issue in learning rate
assert args.epochs>args.lr_warmup_epochs, "Total epochs must be more than warmup epochs"

best_acc1 = 0

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
print("Expr dir", expr_dir)

if not os.path.exists(expr_dir):
    os.makedirs(expr_dir)


# Read MT's baseline checkpoint path
mt_baseline_hash_config = expr_name
mt_baseline_path = os.path.join(mt_root_directory, mt_baseline_hash_config)
ckpt_config = "Checkpoints/checkpoint.pth.tar"
mt_baseline_ckpt_path = os.path.join(mt_baseline_path, ckpt_config)


# Make checkpoints and metrics directory
ckpt_folder = "Checkpoints"
ckpt_dir = os.path.join(expr_dir, ckpt_folder)
metrics_folder = "Metrics"
metrics_dir = os.path.join(expr_dir, metrics_folder)
relu_folder = "ReLUs"
relu_dir = os.path.join(expr_dir, relu_folder)
rbf_folder = "MLP"
rbf_dir = os.path.join(expr_dir, rbf_folder)
adv_datasets_folder = "Adversarial_Datasets"
adv_datasets_dir = os.path.join(expr_dir, adv_datasets_folder)
benign_datasets_folder = "Benign_Datasets"
benign_datasets_dir = os.path.join(expr_dir, benign_datasets_folder)
logs_folder = "Logs"
logs_datasets_dir = os.path.join(expr_dir, logs_folder)
preds_folder = "Predictions"
preds_dir = os.path.join(expr_dir, preds_folder)
# Sub folders for preds_dir
model_preds = "Model"
rbf_preds = "RBF"
perturbed_indices = "Perturbed_Samples"

model_preds_dir = os.path.join(preds_dir, model_preds)
rbf_preds_dir = os.path.join(preds_dir, rbf_preds)
perturbed_indices_dir = os.path.join(preds_dir, perturbed_indices)


if not os.path.exists(relu_dir):
    os.makedirs(relu_dir)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)
if not os.path.exists(rbf_dir):
    os.makedirs(rbf_dir)
if not os.path.exists(adv_datasets_dir):
    os.makedirs(adv_datasets_dir)
if not os.path.exists(benign_datasets_dir):
    os.makedirs(benign_datasets_dir)
if not os.path.exists(preds_dir):
    os.makedirs(preds_dir)
if not os.path.exists(model_preds_dir):
    os.makedirs(model_preds_dir)
if not os.path.exists(rbf_preds_dir):
    os.makedirs(rbf_preds_dir)
if not os.path.exists(perturbed_indices_dir):
    os.makedirs(perturbed_indices_dir)
if not os.path.exists(logs_datasets_dir):
    os.makedirs(logs_datasets_dir)


# Create log files
if(args.evaluate):
    logging_path = os.path.join(expr_dir,"test_log.log")
else:
    logging_path = os.path.join(expr_dir,"train_val_log.log")


logging.basicConfig(filename=logging_path,
                    format='%(asctime)s %(message)s',
                    filemode='w+')
 
# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)



# Create and save YAML file
expr_config_dict = {}
all_args = args._get_kwargs()
expr_config_dict = {tup[0]:tup[1] for tup in all_args}
yaml_file = os.path.join(expr_dir, "Logs/Config.yaml")
with open(yaml_file, 'w') as yaml_out:
    yaml.dump(expr_config_dict, yaml_out)


# Select sparseblock here
sparseblock : Type[Union[SparsifyFiltersLayer, SparsifyKernelGroups]]

if args.sparsefilter == 'SparsifyFiltersLayer':
    sparseblock = SparsifyFiltersLayer
elif args.sparsefilter == 'SparsifyKernelGroups':
    sparseblock = SparsifyKernelGroups

def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(args.seed)

        
    args.distributed = args.multiprocessing_distributed
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        #mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logger.critical(f"Use GPU: {args.gpu} for training")

    # create model
    if args.pretrained:
        logger.critical(f"=> using pre-trained model {args.arch}")        
        if args.arch == 'resnet18':
            model = resnet18(sparsefilter=sparseblock,scale_factor=args.scale_factor)
        if args.new_classifier:
            if args.arch == 'resnet18':
               model.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True)
        if args.arch == 'MLP':
            model = MLP()
    else:
        logger.critical(f"=> creating model {args.arch}")
        if args.arch == 'resnet18':
            model = resnet18(sparsefilter=sparseblock, scale_factor=args.scale_factor)
        if args.new_classifier:
            if args.arch == 'resnet18':
                model.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True)
        if args.arch == 'MLP':
            model = MLP()
    # Add option to freeze/unfreeze more layers
    # TODO
    if args.freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        if args.arch == 'resnet18':
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        logger.critical('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    

    scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )
    """
    warmup_lr_scheduler = LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
    """
    """
    scheduler = SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_scheduler], milestones=[args.lr_warmup_epochs])
    """
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(mt_baseline_ckpt_path):
            print("=> loading checkpoint '{}'".format(mt_baseline_ckpt_path))
            if args.gpu is None:
                checkpoint = torch.load(mt_baseline_ckpt_path)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(mt_baseline_ckpt_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = torch.tensor(best_acc1)
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.critical(f"=> loaded checkpoint '{mt_baseline_ckpt_path}' (epoch {args.start_epoch})")
        else:
            logger.critical(f"=> no checkpoint found at '{mt_baseline_ckpt_path}'")

    random_seed=args.seed
    # GO-GO-GO!
    normalize  =  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    
    """
    
    train_transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
            ])
    """

    """
    train_transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
            ])
    """
    """
    val_transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
            ])
    """
    """
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.RandomErasing(args.random_erasing),
        normalize,
            ])
    """
    """
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    """


    num_classes = args.num_classes
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_categories=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate



    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.RandomErasing(args.random_erasing),
    #normalize,   
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    #normalize,   
    ])

    dataset_path = configs.dataset_root_paths[args.original_dataset]    

    if args.original_dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                                download=False, transform=transform_train)

        valset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                            download=False, transform=transform_test
                                            )

    elif args.original_dataset == 'cifar100':

        trainset = torchvision.datasets.CIFAR100(root=dataset_path, train=True,
                                                download=False, transform=transform_train)

        valset = torchvision.datasets.CIFAR100(root=dataset_path, train=False,
                                            download=False, transform=transform_test
                                            )

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers)
    
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)
        if(((epoch + 1) % args.num_eval_epochs) == 0):
        # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)
            scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()}, epoch, is_best)
            """
            if is_best:
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()}, epoch, is_best)
            # Also save latest checkpoint
            """

    # Test after training
    # Loading the best checkpoint
    best_ckpt_name = "model_best.pth.tar"
    best_ckpt_path = os.path.join(ckpt_dir, best_ckpt_name)
    logger.critical(best_ckpt_path)
    logger.critical("Testing after training")
    if args.gpu is None:
        checkpoint = torch.load(best_ckpt_path, map_location=loc)
    elif torch.cuda.is_available():
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        # Load best checkpoint
        checkpoint = torch.load(best_ckpt_path, map_location=loc)
    best_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    best_acc1 = torch.tensor(best_acc1)
    if args.gpu is not None:
    # best_acc1 may be from a checkpoint from a different GPU
        best_acc1 = best_acc1.to(args.gpu)
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    logger.critical(f"=> loaded checkpoint '{best_ckpt_path}' (epoch {best_epoch})")

    validate(val_loader, model, criterion, args)

    



    ###################################################################################################

def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        if args.weight_repulsion:
            tot_repulsion_loss = 0
            for name, param in model.named_modules():
                if "conv" in name and "layer" in name:
                    kernel_vectors = param.weight.flatten(1)
                    l2_distances = torch.cdist(kernel_vectors, kernel_vectors).triu().unsqueeze(0)
                    l2_blocks = torch.nn.functional.unfold(l2_distances, kernel_size=args.scale_factor, stride=args.scale_factor, padding=0).T
                    step_size = (l2_distances.shape[1] // (args.scale_factor)) + 1
                    extract_indices = torch.arange(0, l2_blocks.shape[0], step=step_size)
                    repulsion_loss = -l2_blocks[extract_indices].sum()
                    tot_repulsion_loss+=repulsion_loss    
        
            # This cannot be a fixed hyperparameter
            tot_repulsion_loss_c = -tot_repulsion_loss.clone().detach()
            #print("Original tot_repulsion_loss_c", tot_repulsion_loss_c)
            lam = 1
            while tot_repulsion_loss_c > loss:
                #print("During loop", tot_repulsion_loss_c)
                tot_repulsion_loss_c = tot_repulsion_loss_c / 10
                lam*=1/10


            tot_repulsion_loss = lam * tot_repulsion_loss
            print(lam)
            #print(f"Loss:{loss}, tot_repulsion_loss: {tot_repulsion_loss}")
            #logger.critical(f"Loss:{loss}, tot_repulsion_loss: {tot_repulsion_loss}")
            loss+=tot_repulsion_loss
        #measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 2))
        #f1_score = compute_f1_score(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            stats = progress.display(i + 1)
            logger.critical(stats)
            #logger.critical(f"F1 score is {f1_score}")


def compute_f1_score(preds, targets):
    pred_classes = torch.argmax(preds, dim=1)
    targets = targets.detach().cpu().numpy()
    pred_classes = pred_classes.detach().cpu().numpy()
    f1_res = f1_score(targets, pred_classes, average='micro')
    return f1_res

def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                
                output= model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 2))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    stats = progress.display(i + 1)
                    logger.critical(stats)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))
    summ_stats = progress.display_summary()
    logger.critical(summ_stats)
    return top1.avg

def save_checkpoint(args, state, epoch, is_best, filename='checkpoint.pth.tar'):
    ckpt_name = f"{filename}"
    ckpt_name = os.path.join(ckpt_dir, ckpt_name)
    torch.save(state, ckpt_name)
    
    if is_best:
        logger.critical("Saving best model")
        best_ckpt_name = "model_best.pth.tar"
        best_ckpt_path = os.path.join(ckpt_dir, best_ckpt_name)
        shutil.copyfile(ckpt_name, best_ckpt_path)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        return ('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))
        return ' '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()
    print("Model trained")