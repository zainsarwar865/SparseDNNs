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
parser.add_argument('--detector_type', type=str)
parser.add_argument('--trainer_type', type=str)
parser.add_argument('--integrated', type=str)
parser.add_argument('--train', type=str)
parser.add_argument('--test_type', type=str)

args = parser.parse_args()

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
import logging
import torch.backends.cudnn as cudnn

# Set seets
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

"""
def load_data(path, label):
    with open(path, 'rb') as iffile:
        features = pickle.load(iffile)
    layer = 'view'
    features = features[layer]
    features_matrix = None

    for i in range(len(features)):
        if isinstance(features_matrix, torch.Tensor):
            features_matrix = torch.cat((features_matrix, features[i]))
        else:
            features_matrix = features[i]

    features = None    
    features_matrix = torch.flatten(features_matrix, start_dim=1, end_dim=-1)
    y = np.empty(features_matrix.shape[0])
    y.fill(label)
    return [features_matrix, y]
"""

def load_data(path, label):
    features_matrix = torch.load(path)
    y = np.empty(features_matrix.shape[0])
    y.fill(label)
    return [features_matrix, y]

    


def unison_shuffled_copies_ind(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]

def unison_shuffled_copies(x_ben, y_ben, x_adv, y_adv):
    X = np.concatenate((x_ben, x_adv), axis=0)
    y = np.concatenate((y_ben, y_adv), axis=0)
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


def get_paths(base_path):
    relu_dir = "ReLUs"
    train_benign = f"ReLUs_{args.attack}_train_benign_{args.total_attack_samples}_detector-type-{args.detector_type}_integrated-{args.integrated}.pth"
    train_adversarial = f"ReLUs_{args.attack}_train_adversarial_{args.total_attack_samples}_detector-type-{args.detector_type}_integrated-{args.integrated}.pth"
    test_benign = f"ReLUs_{args.attack}_test_benign_{args.total_attack_samples}_detector-type-{args.detector_type}_integrated-{args.integrated}.pth"
    test_adversarial = f"ReLUs_{args.attack}_test_adversarial_{args.total_attack_samples}_detector-type-{args.detector_type}_integrated-{args.integrated}.pth"

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

if args.train:

    logging_path = os.path.join(expr_dir,f"train_rbf_{args.detector_type}_{args.total_attack_samples}.log")


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

    train_benign,train_adversarial,test_benign,test_adversarial = get_paths(expr_dir)
    train_benign = load_data(train_benign, 1)
    train_adversarial = load_data(train_adversarial, -1)
    test_benign = load_data(test_benign, 1)
    test_adversarial = load_data(test_adversarial, -1)


    if args.detector_type == 'Quantized':
        print("Quantizing the ReLUs")
        train_benign = quantize(train_benign)
        train_adversarial = quantize(train_adversarial)
        test_benign = quantize(test_benign)
        test_adversarial = quantize(test_adversarial)




    min_train = min(train_benign[0].shape[0], train_adversarial[0].shape[0])
    min_test = min(test_benign[0].shape[0], test_adversarial[0].shape[0])
    # Subsample the larger sets
    rand_indices = np.random.choice(train_benign[0].shape[0], size=min_train, replace=False)
    train_benign[0] = train_benign[0][rand_indices] 
    train_benign[1] = train_benign[1][rand_indices]

    rand_indices = np.random.choice(test_benign[0].shape[0], size=min_test, replace=False)
    test_benign[0] = test_benign[0][rand_indices] 
    test_benign[1] = test_benign[1][rand_indices]

    X_train, y_train = unison_shuffled_copies(train_benign[0], train_benign[1], train_adversarial[0], train_adversarial[1])

    X_test, y_test = unison_shuffled_copies(test_benign[0], test_benign[1], test_adversarial[0], test_adversarial[1])

    if args.detector_type == 'Regular':
        clf = SVC(C=0.7, gamma=0.075)
    elif args.detector_type == 'Quantized':
        clf = SVC(C=0.7, gamma=0.03)

    clf.fit(X_train, y_train)

    # Testing
    x_train_pred = clf.predict(X_train)
    train_acc = len(np.where(x_train_pred == y_train)[0]) / len(X_train)

    x_test_pred = clf.predict(X_test)
    test_acc = len(np.where(x_test_pred == y_test)[0]) / len(X_test)



    logger.critical(f"Training acc : {train_acc}")
    logger.critical(f"Test acc : {test_acc}")

    print(f"Training acc : {train_acc}")
    print(f"Test acc : {test_acc}")


    # Test classwise
    x_train_ben_pred = clf.predict(train_benign[0])
    train_ben_acc = len(np.where(x_train_ben_pred == train_benign[1])[0]) / len(x_train_ben_pred)

    x_train_adv_pred = clf.predict(train_adversarial[0])
    train_adv_acc = len(np.where(x_train_adv_pred == train_adversarial[1])[0]) / len(x_train_adv_pred)

    x_test_ben_pred = clf.predict(test_benign[0])
    test_ben_acc = len(np.where(x_test_ben_pred == test_benign[1])[0]) / len(x_test_ben_pred)

    x_test_adv_pred = clf.predict(test_adversarial[0])
    test_adv_acc = len(np.where(x_test_adv_pred == test_adversarial[1])[0]) / len(x_test_adv_pred)

    logger.critical(f"Training benign acc : {train_ben_acc}")
    logger.critical(f"Training adversarial acc : {train_adv_acc}")
    logger.critical(f"-------------------------------------")
    logger.critical(f"Testing benign acc : {test_ben_acc}")
    logger.critical(f"Testing adversarial acc : {test_adv_acc}")

    print(f"Training benign acc : {train_ben_acc}")
    print(f"Training adversarial acc : {train_adv_acc}")
    print(f"-------------------------------------")
    print(f"Testing benign acc : {test_ben_acc}")
    print(f"Testing adversarial acc : {test_adv_acc}")


    # Save rbf model
    rbf_config = f"RBF_{args.attack}_{args.total_train_samples}_{args.detector_type}.pkl" 
    rbf_path = os.path.join(expr_dir, "RBF", rbf_config)

    with open(rbf_path,'wb') as f:
        pickle.dump(clf,f)
    print("RBF trained...")

else:
    print("Testing RBF...")

    if args.test_type == 'benign':
        logging_path = os.path.join(expr_dir,f"test_rbf_integrated_type-{args.test_type}-detector-type-{args.detector_type}.log")
        _,__,test_path,____ = get_paths(expr_dir)
        print("Test path is ", test_path)
        test_data = load_data(test_path, 1)

    elif args.test_type == 'adversarial':
        print("testing adversarial")
        logging_path = os.path.join(expr_dir,f"test_rbf_integrated_type-{args.test_type}-detector-type-{args.detector_type}.log")
        _,__,___,test_path = get_paths(expr_dir)
        test_data = load_data(test_path, -1)


    if args.detector_type == 'Quantized':
        print("Quantizing the ReLUs")
        test_data = quantize(test_data)

    logging.basicConfig(filename=logging_path,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    
    # Creating an object
    logger = logging.getLogger()
    # Setting the threshold of the logger
    logger.setLevel(logging.INFO)

    # Label
    X, y = unison_shuffled_copies_ind(test_data[0], test_data[1])

    # Loading svm
    rbf_config = f"RBF_{args.attack}_{args.total_train_samples}_{args.detector_type}.pkl"
    rbf_path = os.path.join(expr_dir, "RBF", rbf_config)
    print("Rbf path is ", rbf_path)
    with open(rbf_path,'rb') as in_model:
        clf = pickle.load(in_model)

    # Testing
    x_pred = clf.predict(X)
    acc = len(np.where(x_pred == y)[0]) / len(X)
    logger.critical(f"Acc : {acc}")
    print("Accuracy : ", acc)
    print("RBF tested...")

