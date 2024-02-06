import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--root_hash_config', type=str)
parser.add_argument('--base_dir')
parser.add_argument('--mt_hash_config', type=str)
parser.add_argument('--seed', type=int, help="seed for pandas sampling")
parser.add_argument('--gpu', type=int)
parser.add_argument('--total_attack_samples', type=int)
parser.add_argument('--total_train_samples', type=int)
parser.add_argument('--attack', type=str)
parser.add_argument('--attack_split', type=str)
parser.add_argument('--detector_type', type=str)
parser.add_argument('--trainer_type', type=str)
parser.add_argument('--integrated', type=str)
parser.add_argument('--train', type=str)
parser.add_argument('--test_type', type=str)
parser.add_argument('--c', type=float)
parser.add_argument('--d', type=float)
parser.add_argument('--hidden_layers', type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--lr', dest='lr', type=float)
parser.add_argument('--lr_min', default=0.0, type=float)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--weight_decay',  type=float)
parser.add_argument('--print_freq', default=300, type=int)
parser.add_argument('--evaluate', dest='evaluate', default=False)
parser.add_argument('--num_eval_epochs', type=int, default=1)


args = parser.parse_args()

args.distributed = False

if args.integrated == "True":
    args.integrated = True
else:
    args.integrated = False

if args.train == "True":
    args.train = True
else:
    args.train = False

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import random
import shutil
import time
import warnings
from enum import Enum
import yaml
from collections import defaultdict
import pickle
import torch
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
import torchvision
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import hashlib
import torch.backends.cudnn as cudnn
from utils.dataset import CustomImageDataset_Adv
from utils.MLP import MLP



###########
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
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Subset
import hashlib
from sklearn.metrics import f1_score
import logging
from collections import Counter
from utils.transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
import utils.utils as utils
import utils.configs as configs

###########

# Set seeds
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

def load_data(path, label, flipped_indices=None):
    features_matrix = torch.load(path)
    if isinstance(flipped_indices, torch.Tensor):
        features_matrix = features_matrix[flipped_indices]
        # Merge batch and pixel dimensions
    features_matrix  = features_matrix.flatten(0,1)
    y = np.empty(features_matrix.shape[0])
    y.fill(label)
    return [features_matrix, y]

def unison_shuffled_copies_ind(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X, y
    
def unison_shuffled_copies(x_ben, y_ben, x_adv, y_adv):
    X = np.concatenate((x_ben, x_adv), axis=0)
    y = np.concatenate((y_ben, y_adv), axis=0)
    y = np.int32(y)
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


def get_paths(base_path):
    relu_dir = "ReLUs"
    train_benign = f"ReLUs_{args.attack}_train_benign_{args.total_attack_samples}_detector-type-{args.detector_type}_integrated-{args.integrated}_c-{args.c}_d-{args.d}.pth"
    train_adversarial = f"ReLUs_{args.attack}_train_adversarial_{args.total_attack_samples}_detector-type-{args.detector_type}_integrated-{args.integrated}_c-{args.c}_d-{args.d}.pth"
    test_benign = f"ReLUs_{args.attack}_test_benign_{args.total_attack_samples}_detector-type-{args.detector_type}_integrated-{args.integrated}_c-{args.c}_d-{args.d}.pth"
    test_adversarial = f"ReLUs_{args.attack}_test_adversarial_{args.total_attack_samples}_detector-type-{args.detector_type}_integrated-{args.integrated}_c-{args.c}_d-{args.d}.pth"

    train_benign = os.path.join(base_path, relu_dir, train_benign)
    train_adversarial = os.path.join(base_path, relu_dir,  train_adversarial)
    test_benign = os.path.join(base_path,  relu_dir, test_benign)
    test_adversarial = os.path.join(base_path,  relu_dir, test_adversarial)

    all_paths = [train_benign,train_adversarial,test_benign,test_adversarial]

    return all_paths


def quantize(X):
    X[0] = torch.where(X[0] > 0, torch.ones_like(X[0]), torch.zeros_like(X[0]))
    return X

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
ckpt_dir = os.path.join(expr_dir, "MLP")


if args.train:

    logging_path = os.path.join(expr_dir,f"Logs/train_mlp_detector-{args.detector_type}_{args.total_attack_samples}_Layers-{args.hidden_layers}.log")
else:
    logging_path = os.path.join(expr_dir,f"Logs/test_mlp_integrated-{args.integrated}_type-{args.test_type}-detector-type-{args.detector_type}_c-{args.c}_d-{args.d}_Layers-{args.hidden_layers}.log")

logging.basicConfig(filename=logging_path,
                        format='%(asctime)s %(message)s',
                        filemode='w')
# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

if args.gpu is not None:
    logger.critical(f"Use GPU: {args.gpu} for training")


if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    logger.critical('using CPU, this will be slow')


def main():
    
    # Train code
    ########################################################################################################
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(args.seed)

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    if args.train:  
        main_worker_train(args.gpu, ngpus_per_node, args)
    else:
        main_worker_test(args.gpu, ngpus_per_node, args)

def main_worker_train(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = 0

    ##########################################################################################

    # Compute filtered accuracy
    flipped_indices_config = f"Predictions/Perturbed_Samples/{args.attack}_benign_samples_{args.total_attack_samples}_test_detector-type-{args.detector_type}_integrated-{args.integrated}_c-{args.c}_d-{args.d}.pt"
    flipped_indices_path = os.path.join(expr_dir, flipped_indices_config)
    flipped_indices = torch.load(flipped_indices_path)

    # create model
    model = MLP()

    # Create and save YAML file
    expr_config_dict = {}
    all_args = args._get_kwargs()
    expr_config_dict = {tup[0]:tup[1] for tup in all_args}
    yaml_file = os.path.join(expr_dir, "Logs/MLP_train_Config.yaml")
    with open(yaml_file, 'w') as yaml_out:
        yaml.dump(expr_config_dict, yaml_out)

    train_benign,train_adversarial,test_benign,test_adversarial = get_paths(expr_dir)
    train_benign = load_data(train_benign, 0)
    train_adversarial = load_data(train_adversarial, 1)
    test_benign = load_data(test_benign, 0, flipped_indices=flipped_indices)
    test_adversarial = load_data(test_adversarial, 1, flipped_indices=flipped_indices)


    if args.detector_type == 'Quantized':
        print("Quantizing the ReLUs")
        train_benign = quantize(train_benign)
        train_adversarial = quantize(train_adversarial)
        test_benign = quantize(test_benign)
        test_adversarial = quantize(test_adversarial)

    X_train, y_train = unison_shuffled_copies(train_benign[0], train_benign[1], train_adversarial[0], train_adversarial[1])
    x_y_train = [X_train, y_train]
    trainset = CustomImageDataset_Adv(x_y_train)

    X_test, y_test = unison_shuffled_copies(test_benign[0], test_benign[1], test_adversarial[0], test_adversarial[1])
    x_y_test = [X_test, y_test]

    valset = CustomImageDataset_Adv(x_y_test)
    ##########################################################################################

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device("cpu")

    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)
        model = model.cuda(device)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    

    scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )

    random_seed=args.seed
    # GO-GO-GO
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)

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
            if is_best:
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'arch': "mlp",
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()}, epoch, is_best)


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


    logger.critical(f"Validating entire dataset")
    validate(val_loader, model, criterion, args)

    logger.critical(f"Validating benign dataset")

    #X_test, y_test = unison_shuffled_copies(test_benign[0], test_benign[1], test_adversarial[0], test_adversarial[1])
    x_y_test = [test_benign[0], test_benign[1]]

    valset = CustomImageDataset_Adv(x_y_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers)


    validate(val_loader, model, criterion, args)

    logger.critical(f"Validating adversarial dataset")

    x_y_test = [test_adversarial[0], test_adversarial[1]]

    valset = CustomImageDataset_Adv(x_y_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers)

    validate(val_loader, model, criterion, args)



    #####################################################################################################################








###################################################

def main_worker_test(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = 0

    ##########################################################################################

    # Compute filtered accuracy
    flipped_indices_config = f"Predictions/Perturbed_Samples/{args.attack}_benign_samples_{args.total_attack_samples}_test_detector-type-{args.detector_type}_integrated-{args.integrated}_c-{args.c}_d-{args.d}.pt"
    flipped_indices_path = os.path.join(expr_dir, flipped_indices_config)
    flipped_indices = torch.load(flipped_indices_path)

    # create model
    model = MLP()
    model = model.eval()
    
    if args.test_type == 'benign':
        _,__,test_path,____ = get_paths(expr_dir)
        test_data = load_data(test_path, 0, flipped_indices=flipped_indices)

    elif args.test_type == 'adversarial':
        _,__,___,test_path = get_paths(expr_dir)
        test_data = load_data(test_path, 1, flipped_indices=flipped_indices)


    if args.detector_type == 'Quantized':
        print("Quantizing the ReLUs")
        test_data = quantize(test_data)

    X, y = unison_shuffled_copies_ind(test_data[0], test_data[1])

    x_y_test = [X, y]
    valset = CustomImageDataset_Adv(x_y_test)


    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device("cpu")



    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)
        model = model.cuda(device)


    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    random_seed=args.seed
    # GO-GO-GO
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])
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
    logger.critical(f"=> loaded checkpoint '{best_ckpt_path}' (epoch {best_epoch})")

    logger.critical(f"Validating test dataset")
    predictions = {}

    acc, all_preds, all_preds_scores = validate_per_class(val_loader, model, criterion, args)

    predictions['true_labels'] = y
    predictions['pred_labels'] = all_preds
    predictions['pred_scores'] = all_preds_scores

    # Process Image level classification decisions
    sampled_dimension = 50

    true_labels = predictions['true_labels']
    pred_labels = predictions['pred_labels']
    sampled_dimension = 50
    benign_votes = [0 for i in range(len(true_labels) // sampled_dimension)]
    adversarial_votes = [0 for i in range(len(true_labels) // sampled_dimension)]


    pred_labels = np.asarray(pred_labels)
    for idx, i in enumerate(range(0, len(true_labels), sampled_dimension)):
        curr_pred = pred_labels[i:i+sampled_dimension]
        adv_votes = np.count_nonzero(curr_pred == 1)
        ben_votes = np.count_nonzero(curr_pred == 0)
        adversarial_votes[idx] = adv_votes
        benign_votes[idx] = ben_votes

    outputs = np.stack((benign_votes, adversarial_votes))
    outputs = torch.from_numpy(outputs)
    preds = torch.argmax(outputs, dim=0)    
    y_images = torch.tensor(y[0:len(preds)])
    # Img acc
    img_acc = sum(torch.eq(preds, y_images).to(dtype=int)) / len(preds)
    predictions['true_labels_images'] = y_images
    predictions['pred_labels_images'] = preds
    predictions['acc_images'] = img_acc
    print("Image level accuracy is : ", img_acc)
    logger.critical(f"Image level accuracy is : {img_acc}" )



    # Save RBF preds
    preds_config = f"Predictions/RBF/{args.attack}_type-{args.test_type}_{args.total_attack_samples}_test_detector-type-{args.detector_type}_integrated-{args.integrated}_rbf_c-{args.c}_d-{args.d}.pickle"
    preds_path = os.path.join(expr_dir, preds_config)
    with open(preds_path, 'wb') as o_file:
        pickle.dump(predictions, o_file)


    logger.critical(f"Validating filtered dataset")

    #X_test, y_test = unison_shuffled_copies(test_benign[0], test_benign[1], test_adversarial[0], test_adversarial[1])

    # Flipped indices
    test_data[0] = test_data[0][flipped_indices]
    test_data[1] = test_data[1][flipped_indices]
    x_y_test = [test_data[0], test_data[1]]
    valset = CustomImageDataset_Adv(x_y_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers)
    validate(val_loader, model, criterion, args)



###################################################









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
        target = target.type(torch.LongTensor)   # casting to long

        data_time.update(time.time() - end)
        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
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


def save_checkpoint(args, state, epoch, is_best, filename='checkpoint.pth.tar'):

    ckpt_name = "model_best.pth.tar"
    ckpt_name = os.path.join(ckpt_dir, ckpt_name)
    torch.save(state, ckpt_name)
    """
    if is_best:
        logger.critical("Saving best model")
        best_ckpt_name = "model_best.pth.tar"
        best_ckpt_path = os.path.join(ckpt_dir, best_ckpt_name)
        shutil.copyfile(filename, best_ckpt_path)
    """
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


def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                target = target.type(torch.LongTensor)   # casting to long
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




def validate_per_class(val_loader, model, criterion, args):
    per_instance_preds = Counter()

    def run_validate(loader, base_progress=0):
        all_preds = []
        all_preds_scores = None
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                target = target.type(torch.LongTensor)   # casting to long
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
                pred_classes = torch.argmax(output, dim=1)
                pred_classes = pred_classes.detach().cpu().numpy().tolist()
                labels = target.detach().cpu().numpy().tolist()
                per_instance_preds.update(pred_classes)
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
                                
                all_preds.extend(pred_classes)
                if isinstance(all_preds_scores, torch.Tensor):
                    all_preds_scores = torch.concat((all_preds_scores, output.detach().cpu()), dim=0)
                else:
                    all_preds_scores = output.detach().cpu()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    stats = progress.display(i + 1)
                    logger.info(stats)
        return all_preds, all_preds_scores
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

    all_preds, all_preds_scores = run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    summ_stats = progress.display_summary()
    logger.info(summ_stats)
    return top1.avg, all_preds, all_preds_scores


if __name__ == '__main__':
    main()
    print("Model trained and tested....")