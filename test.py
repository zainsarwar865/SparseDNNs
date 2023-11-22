import argparse
import os
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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision
import torchvision.models as models
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Subset
from utils.dataset import CustomImageDataset, CustomImageDataset_Adv
import hashlib
from sklearn.metrics import f1_score
import logging
from collections import Counter
from utils.transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
import utils.utils as utils
import utils.configs as configs
#from torchvision.models.feature_extraction import create_feature_extractor


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--root_hash_config', type=str)
parser.add_argument('--base_dir')
parser.add_argument('--mt_hash_config', type=str)
parser.add_argument('--arch', metavar='ARCH',)
parser.add_argument('--num_eval_epochs', type=int)
parser.add_argument('--workers', default=4, type=int)
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
parser.add_argument('--print_freq', default=300, type=int)
parser.add_argument('--test', type=str)
parser.add_argument('--evaluate', dest='evaluate', default=False)
parser.add_argument('--pretrained',type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--multiprocessing_distributed',type=bool, default=False)
parser.add_argument('--seed', type=int, help="seed for pandas sampling")
parser.add_argument('--freeze_layers', type=str)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--trainer_type', type=str)
parser.add_argument('--original_dataset', type=str)
parser.add_argument('--original_config', type=str)
parser.add_argument('--new_classifier', type=str)
parser.add_argument('--test_adversarial', type=str)
parser.add_argument('--attack', type=str)
parser.add_argument('--attack_split', type=str)
parser.add_argument('--total_attack_samples', type=int)


args = parser.parse_args()

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


if args.test_adversarial == "True":
    args.test_adversarial = True
else:
    args.test_adversarial = False


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


if not os.path.exists(expr_dir):
    os.makedirs(expr_dir)

# Make checkpoints and metrics directory
ckpt_folder = "Checkpoints"
ckpt_dir = os.path.join(expr_dir, ckpt_folder)
metrics_folder = "Metrics"
metrics_dir = os.path.join(expr_dir, metrics_folder)
relu_folder = "ReLUs"
relu_dir = os.path.join(expr_dir, relu_folder)

if not os.path.exists(relu_dir):
    os.makedirs(relu_dir)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)


# Create log files
logging_path = os.path.join(expr_dir,"test_log.log")


logging.basicConfig(filename=logging_path,
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

# Create and save YAML file
expr_config_dict = {}
all_args = args._get_kwargs()
expr_config_dict = {tup[0]:tup[1] for tup in all_args}
yaml_file = os.path.join(expr_dir, "Config.yaml")
with open(yaml_file, 'w') as yaml_out:
    yaml.dump(expr_config_dict, yaml_out)

print("Starting test...")

def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    args.distributed = args.multiprocessing_distributed
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
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
            model = models.__dict__[args.arch](weights=ResNet18_Weights.IMAGENET1K_V2)
        if args.arch == 'resnet50':
            model = models.__dict__[args.arch](weights=ResNet50_Weights.IMAGENET1K_V2)
        if args.new_classifier:
            if args.arch == 'resnet50':
               model.fc = nn.Linear(in_features=2048, out_features=args.num_classes, bias=True)
            if args.arch == 'resnet18':
               model.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True)
    else:
        logger.critical(f"=> creating model {args.arch}")
        model = models.__dict__[args.arch]()
        
        if args.new_classifier:
            if args.arch == 'resnet50':
                model.fc = nn.Linear(in_features=2048, out_features=args.num_classes, bias=True)
            if args.arch == 'resnet18':
                model.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True)
    
    
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
    random_seed=args.seed
    # GO-GO-GO!
    normalize  =  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))    


    num_classes = args.num_classes

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,   
    ])

    

    if args.test_adversarial:
        # Load adversarial dataset
        adv_dataset_config = f"Adversarial_Datasets/{args.attack}_adv_samples_{args.total_attack_samples}_{args.attack_split}.pickle"
        adv_dataset_path = os.path.join(expr_dir, adv_dataset_config)

        with open(adv_dataset_path, 'rb') as adv_set:
            adv_samples = pickle.load(adv_set)

        valset = CustomImageDataset_Adv(adv_samples)
    else:
        valset = torchvision.datasets.CIFAR10(root='/bigstor/zsarwar/CIFAR10', train=False,
                                            download=False, transform=transform_test
                                            )
        
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers)
    
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
    logger.critical(f"Evaluating {args.attack_split}")
    validate(val_loader, model, criterion, args)




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
                output = model(images)
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
    print("Model tested")