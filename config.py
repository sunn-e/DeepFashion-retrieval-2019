# -*- coding: utf-8 -*-
"""
# backup for reference. 
#This is for Nvidia titan X. :p Gives out of memory for 1060. Gotta reduce the sizes

GPU_ID = 0
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
TRIPLET_BATCH_SIZE = 3
EXTRACT_BATCH_SIZE = 128
TEST_BATCH_COUNT = 30
NUM_WORKERS = 4
LR = 0.001
MOMENTUM = 0.5
EPOCH = 10
DUMPED_MODEL = "/home/l/Dataset/model_10_final.pth.tar"

LOG_INTERVAL = 10
DUMP_INTERVAL = 500
TEST_INTERVAL = 100

DATASET_BASE = r'/home/l/Dataset/DeepFashion/Category and Attribute Prediction Benchmark'
ENABLE_INSHOP_DATASET = True
INSHOP_DATASET_PRECENT = 0.8
IMG_SIZE = 256
CROP_SIZE = 224
INTER_DIM = 512
CATEGORIES = 20
N_CLUSTERS = 50
COLOR_TOP_N = 10
TRIPLET_WEIGHT = 2.0
ENABLE_TRIPLET_WITH_COSINE = False  # Buggy when backward...
COLOR_WEIGHT = 0.1
DISTANCE_METRIC = ('euclidean', 'euclidean')
FREEZE_PARAM = False

"""

#Sunny's version for running on gtx 1060. Goodluck with time though :p
#please dont forget to copy cudnn files under respective usr/local/cuda folders. You may use below files when not able to copy cudnn files.

# command to change folder permissions 'sudo chmod -R a+rwx /usr/local/cuda/lib64' and 
# sudo chmod -R a+rwx /usr/local/cuda/include


GPU_ID = 0
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
TRIPLET_BATCH_SIZE = 1
EXTRACT_BATCH_SIZE = 64
TEST_BATCH_COUNT = 15
NUM_WORKERS = 4
LR = 0.001
MOMENTUM = 0.5
EPOCH = 10
DUMPED_MODEL = "/home/l/Dataset/model_10_final.pth.tar"

LOG_INTERVAL = 10
DUMP_INTERVAL = 500
TEST_INTERVAL = 100

DATASET_BASE = r'/home/l/Dataset/DeepFashion/Category and Attribute Prediction Benchmark'
ENABLE_INSHOP_DATASET = True
INSHOP_DATASET_PRECENT = 0.8
IMG_SIZE = 256
CROP_SIZE = 224
INTER_DIM = 512
CATEGORIES = 20
N_CLUSTERS = 20 # used in kmeans.py
COLOR_TOP_N = 10
TRIPLET_WEIGHT = 2.0
ENABLE_TRIPLET_WITH_COSINE = False  # Buggy when backward...
COLOR_WEIGHT = 0.1
DISTANCE_METRIC = ('euclidean', 'euclidean')
FREEZE_PARAM = False
