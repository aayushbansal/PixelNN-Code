% compute nearest neighbors for the imgs in the 
% val set. Use the imgs from train set...
clc; clear all;

% read the list of query images ---
CACHENAME = 'edges-to-shoes';


VAL_DATA_PATH = ['./cachedir/', CACHENAME, '/feats/val/'];
val_img_list = dir([VAL_DATA_PATH, '*.mat']);
val_img_list = {val_img_list.name};
val_img_list = val_img_list';


DUMP_DATA_PATH = ['./cachedir/', CACHENAME,'/nns_conv5_from_handbags/'];
if(~isdir(DUMP_DATA_PATH))
	mkdir(DUMP_DATA_PATH);
end

% list of the images in the train set --
TRAIN_DATA_PATH = ['./cachedir/edges-to-bags/feats/train/'];
train_img_list = dir([TRAIN_DATA_PATH, '*.mat']);
train_img_list = {train_img_list.name};
train_img_list = train_img_list';

% for each query img, 
for i = 1:length(val_img_list)

        display(['Query Img: ', val_img_list{i}]);
   		
	% check if file already exists 
	if(exist([DUMP_DATA_PATH, val_img_list{i}], 'file'))
		continue;
	end


        ith_feat = load([VAL_DATA_PATH,val_img_list{i}], 'c5');
	ith_feat = ith_feat.c5(:);
	ith_feat_norm = sqrt(ith_feat'*ith_feat) + eps;
	ith_feat = ith_feat/ith_feat_norm;		
	ith_det_scores = zeros(length(train_img_list),1);
	
	% compute the cosine distance for the dataset in 
	% the train dataset --
	tic;
	for j = 1:length(train_img_list)

		% load the file 
		display(['Searching through: ', train_img_list{j}]);
		jth_file = load([TRAIN_DATA_PATH, ...
				  train_img_list{j}], 'c5');
		jth_feat = jth_file.c5(:);
		jth_feat = jth_feat/(sqrt(jth_feat'*jth_feat)+eps);
		ith_det_scores(j) = ith_feat'*jth_feat;	

	end	
	toc;

	% save the file --
	det_scores = ith_det_scores;
	img_list = train_img_list;
	save([DUMP_DATA_PATH, val_img_list{i}],...
				 'det_scores', 'img_list');
end
