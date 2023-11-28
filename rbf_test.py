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
parser.add_argument('--trainer_type', type=str)


args = parser.parse_args()

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
from sklearn.svm import SVC
import hashlib
import logging

np.random.seed(args.seed)

def load_data(path, label):
    with open(path, 'rb') as iffile:
        features = pickle.load(iffile)
    layer = 'flatten'
    features = features[layer]
    features_matrix = None

    for i in range(len(features)):
        if isinstance(features_matrix, torch.Tensor):
            features_matrix = torch.cat((   features_matrix, features[i]))
        else:
            features_matrix = features[i]

    features = None    
    features_matrix = torch.flatten(features_matrix, start_dim=1, end_dim=-1)
    y = np.empty(features_matrix.shape[0])
    y.fill(label)
    return [features_matrix, y]



def unison_shuffled_copies(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


def get_paths(base_path):
    relu_dir = "ReLUs"
    config = f"ReLUs_{args.attack}_{args.attack_split}_adversarial_{args.total_attack_samples}_integrated-True.pkl"
    config_path = os.path.join(base_path, relu_dir, config)
    return config_path

print("Testing the RBF...")

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

logging_path = os.path.join(expr_dir,"train_val_rbf_post_attack.log")

logging.basicConfig(filename=logging_path,
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

test_path = get_paths(expr_dir)

# Label
test_data = load_data(test_path, -1)
X, y = unison_shuffled_copies(test_data[0], test_data[1])

# Loading svm
rbf_config = f"RBF_{args.attack}_{args.total_train_samples}.pkl" 
rbf_path = os.path.join(expr_dir, "RBF", rbf_config)

with open(rbf_path,'rb') as in_model:
    clf = pickle.load(in_model)

# Testing
x_pred = clf.predict(X)
acc = len(np.where(x_pred == y)[0]) / len(X)
logger.critical(f"Acc : {acc}")
print("Accuracy : ", acc)
print("RBF tested...")

