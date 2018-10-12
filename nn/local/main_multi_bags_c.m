% this is the main script to generate multiple outputs
% for a given input-output pair --
clc; clear all;

GPU = 1;

% Assign all the required paths for the task - 
CACHENAME = 'edges-to-bags';
REQ_PATHS.VAL_DATA_PATH  = ['./cachedir/', CACHENAME, '/feats/val/'];
REQ_PATHS.VAL_IMG_PATH = ['./cachedir/',CACHENAME, '/val/'];
REQ_PATHS.TRAIN_DATA_PATH = ['./cachedir/', CACHENAME, '/feats/train/'];
REQ_PATHS.TRAIN_IMG_ORG_PATH = ['./cachedir/', CACHENAME, '/train/'];
REQ_PATHS.TRAIN_IMG_PATH = ['./cachedir/', CACHENAME, '/train/'];
REQ_PATHS.CACHE_DIR = ['./cachedir/', CACHENAME,'/nns_conv5/'];
REQ_PATHS.DUMP_DATA_PATH = ['./cachedir/', CACHENAME, '/hi-res-multi-color/'];
REQ_PATHS.MODEL_DIR = ['./data/net/vgg16_seg_hypercol/'];
REQ_PATHS.DEPLOY_FILE = [REQ_PATHS.MODEL_DIR 'deploy.prototxt'];;
REQ_PATHS.MODEL_WEIGHTS = [REQ_PATHS.MODEL_DIR, 'vgg16_seg_hypercol.caffemodel'];
REQ_PATHS.RM_STRING = [''];
REQ_PATHS.PIX_PATCH = ['/mnt/ssd1/users/aayushb/PixelNN/nn/nn/'];

% generate multiple outputs -- 
generate_multi_outputs_bags_c(REQ_PATHS, GPU)

