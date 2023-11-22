import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from collections import defaultdict
import pickle
import torch
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
import torchvision
with open("/bigstor/zsarwar/SparseDNNs/MT_CIFAR10_full_10_d5f3f545a0883adb9c8f98e2a6ba4ac7/MT_Baseline_d2a45a4dd02a5e037e5954b82387e666/ReLUs/ReLUs_Clean.pkl", 'rb') as iffile:
    features = pickle.load(iffile)
features.keys()
layer = 'layer_4_0_1'
features = features[layer]
features_matrix = None

for i in range(len(features)):
    if isinstance(features_matrix, torch.Tensor):
        features_matrix = torch.cat((features_matrix, features[i]))
    else:
        features_matrix = features[i]

features = None
features_matrix = torch.flatten(features_matrix, start_dim=1, end_dim=-1)
# Quantize
# Comput per-dimension mean
dim_mean = features_matrix.mean(dim=0)
binarized_matrix = torch.where(features_matrix >= dim_mean, torch.ones_like(features_matrix), torch.zeros_like(features_matrix))
features_matrix = None

torch.cuda.empty_cache()
binarized_matrix = binarized_matrix.cpu().numpy()


model = OneClassSVM(kernel='rbf')
model.fit(binarized_matrix)
# save
#with open('/bigstor/zsarwar/SparseDNNs/MT_CIFAR10_full_10_d5f3f545a0883adb9c8f98e2a6ba4ac7/MT_Baseline_d2a45a4dd02a5e037e5954b82387e666/RBF/rbf_detector.pkl','wb') as f:
    #pickle.dump(model,f)
with open('/bigstor/zsarwar/SparseDNNs/MT_CIFAR10_full_10_d5f3f545a0883adb9c8f98e2a6ba4ac7/MT_Baseline_d2a45a4dd02a5e037e5954b82387e666/RBF/rbf_detector.pkl', 'rb') as f:
    model = pickle.load(f)
test_data = binarized_matrix[0:1000, :]
res = model.predict(test_data)
np.where(res == 1)[0].shape