% define startup:
% add the list of the paths for the startup
addpath(genpath('tools/caffe'));
addpath(genpath('./experiments/faces/'));
addpath('./nn/shoesandbags/');
addpath('./nn/nn/');
addpath('./utils/');
addpath('./experiments/eval/');

% following are the set of options that need to be 
% set to use the train code --
%options.cachepath = ['./cachedir/'];
%options.datapath = ['./data/'];
%options.cnn_input_size = 256;
%options.segimbatch = 5;
%options.segsamplesize = 2000;
%options.segbatchsize = (options.segimbatch)*(options.segsamplesize);
%options.trainFlip = 1;
%options.seed = 1989;
%options.segepoch = 80;
%options.saveEpoch = 1;
%options.meanvalue = [98.0035, 71.5392, 100.4979];% for cityscape
%options.meanvalue = [190.2004, 77.5135, 56.5447]; % for facades



% options for face surface normals --
options.cachepath = ['./cachedir/'];
options.datapath = ['./data/'];
options.cnn_input_size = 96;
options.labelsimbatch = 5;
options.labelssamplesize = 2000;
options.labelsbatchsize = (options.labelsimbatch)*(options.labelssamplesize);
options.trainFlip = 1;
options.seed = 1989;
options.labelsepoch = 80;
options.saveEpoch = 1;
options.meanvalue = [102.9801, 115.9465, 122.7717];% for cityscape 
