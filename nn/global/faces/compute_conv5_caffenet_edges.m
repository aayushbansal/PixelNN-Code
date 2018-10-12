% compute fc-7 features for the images
clc; clear all;

% define cachedir --
CACHENAME = 'faces-edges';
SPLIT_N = 'val_vgg';
CACHEDIR = ['./cachedir/', CACHENAME, '/feats/', SPLIT_N, '/'];
if(~isdir(CACHEDIR))
	mkdir(CACHEDIR);
end

% load the list of images
load('./data/faces/edges_val.mat');

%imagelist = strrep(edgeslist, '.jpg', '.png');
imagelist = edgeslist;
imagelist_r = strcat('/mnt/pcie1/user/aayushb/Faces/',...
				imagelist);
imagelist_w =   strrep(imagelist, '/', '-');

% set gpu --
caffe.set_mode_gpu();
gpu_id = 0;  
caffe.set_device(gpu_id);

% load neural network in memory --
model_dir = './data/net/faces/';
net_model = [model_dir 'deploy_nn.prototxt'];
net_weights = ['./cachedir/faces-edges/TRAIN/(03).caffemodel'];
phase = 'test'; 

%mean_data = load([model_dir, 'caffenet_mean.mat'],'image_mean');
%mean_data = mean_data.image_mean;
%mean_data = imresize(mean_data, [96,96]);
cnn_input_size = 96;
mean_data = cat(3,  102.9801*ones(cnn_input_size),...
                     115.9465*ones(cnn_input_size),...
                     122.7717*ones(cnn_input_size));

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

% for each image --
for i = 1:length(imagelist)

	ith_image_fname = imagelist_r{i};
	ith_image_sname = imagelist_w{i};
	display(['Image ID:  ' , ith_image_sname]);
	

	if(exist([CACHEDIR, '/',....
		  strrep(ith_image_sname, '.png', '.mat')], 'file'))
		continue;
	end

	% read the image --
	im = imread(ith_image_fname);
	if(size(im,3)~=3)
		im = cat(3, im, im, im);
	end

	im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
	im_data = permute(im_data, [2, 1, 3]);  % flip width and height
	im_data = single(im_data);  % convert from uint8 to single
	im_data = im_data - mean_data;  

	% feed it forward to the CNN
	net_data(:,:,:,1) = im_data;
	c5_feat = net.forward({net_data});
	c5 = c5_feat{1};

	% dump it in the disk
	save_file_name = [CACHEDIR,  '/',....
                          strrep(ith_image_sname, '.png', '.mat')];
	save(save_file_name, 'c5');
	
end	

caffe.reset_all;
