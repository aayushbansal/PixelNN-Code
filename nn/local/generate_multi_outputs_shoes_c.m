% visualize the correspondences for the faces
% using the hypercolumn features trained for 
% segmentation --
function generate_multi_outputs_shoes_c(REQ_PATHS, GPU)

% list of query images --
VAL_DATA_PATH = REQ_PATHS.VAL_DATA_PATH;

% load the list of images --
val_img_list = dir([VAL_DATA_PATH, '*.mat']);
val_img_list = {val_img_list.name};
val_img_list = val_img_list';

TRAIN_IMG_ORG_PATH = REQ_PATHS.TRAIN_IMG_ORG_PATH;
TRAIN_IMG_PATH = REQ_PATHS.TRAIN_IMG_PATH;
VAL_IMG_PATH = REQ_PATHS.VAL_IMG_PATH;

% inputs all the variations;
list_variations;

CACHE_DIR = REQ_PATHS.CACHE_DIR;
DUMP_DATA_PATH = REQ_PATHS.DUMP_DATA_PATH;
if(~isdir(DUMP_DATA_PATH))
        mkdir(DUMP_DATA_PATH);
end

% initialize the network --
% set gpu --
caffe.set_mode_gpu();
gpu_id = GPU;
caffe.set_device(gpu_id);

% load neural network in memory --
model_dir = REQ_PATHS.MODEL_DIR;
net_model = REQ_PATHS.DEPLOY_FILE;
net_weights = REQ_PATHS.MODEL_WEIGHTS;
phase = 'test';

mean_data = load([model_dir, 'mean_file.mat'],'mean_data');
mean_data = mean_data.mean_data;
mean_data = mean_data(1:96,1:96,:);

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

% for each query img --
for i = 1:length(val_img_list)

        ith_image_name = strrep(val_img_list{i}, '.mat', '');

        display(['Image ID: ', ith_image_name]);

	if(~isdir([DUMP_DATA_PATH, '/', ith_image_name, '/']))
		mkdir([DUMP_DATA_PATH, '/', ith_image_name, '/']);
	end

        if(exist([DUMP_DATA_PATH,  '/', ith_image_name, '/'....
                   ith_image_name, '_', num2str(size(nn_input,1), '%03d') '.png'], 'file'))
                continue;
        end

	if(isLocked([DUMP_DATA_PATH,'/', ith_image_name, '/', ith_image_name]))
                continue;
        end

	% -- 
        % read the image --
        im = imread([VAL_IMG_PATH,ith_image_name, '.png']);

	if(size(im,3)~=3)
		im = cat(3, im, im, im);
	end

	[f2_sel] = compute_hypercolumn_feat(im, mean_data, net);
	[x_sft, y_sft] = meshgrid(1:1:size(f2_sel,1), 1:1:size(f2_sel,1));
         y_sft = y_sft(:); x_sft = x_sft(:);

	% normalize the data --	
	f2_sel_norm = repmat(sqrt(sum(f2_sel.*f2_sel,2)),...
                                 1, size(f2_sel,2));	
	f2_sel = f2_sel./(f2_sel_norm+eps);
	f2_sel = reshape(f2_sel, [size(f2_sel,1)*size(f2_sel,2),size(f2_sel,3)]);

	% -- 
	% get the nearest neighbor data --
        ith_val_data = load([CACHE_DIR, ith_image_name, '.mat'], ...
                        'det_scores', 'img_list');
        ith_det = ith_val_data.det_scores;
        ith_train_il = ith_val_data.img_list;

	ith_outputs = cell(size(nn_input,1),1);

	for j = 1:size(nn_input,1)

	        [~,I] = sort(ith_det, 'descend');
       		I = I(nn_input(j,2):nn_input(j,1)+nn_input(j,2)-1);
	        jth_det = ith_det(I);
        	jth_train_il = ith_train_il(I);

		% dump the NN-train hypercol data
		nn_pixel_scor = zeros(size(f2_sel,1),1);
	        nn_pixel_loxy = zeros(size(f2_sel,1),2);
		nn_pixel_NNid = zeros(size(f2_sel,1),1);
		nn_pixel_ldata = zeros(size(f2_sel,1),3);
		nn_pixel_lscore = zeros(size(f2_sel,1),1);	
	
		% iterate for the set of nearest neighbors -
		for k = 1:nn_input(j,1)
			
			k_pix_patch = nn_input(j,3);
			if(k_pix_patch > 1)
				if(exist([REQ_PATHS.PIX_PATCH,...
					 '/patch_', num2str(k_pix_patch,'%02d'),...
					 '.mat'], 'file'))
					load([REQ_PATHS.PIX_PATCH,...
                                         '/patch_', num2str(k_pix_patch,'%02d'),...
                                         '.mat'], 'img_pat_loc')
				else
					error('generate the k-pix-patch-first');
				end
			end

                	% read the required image --
			display(['Nearest Neighbor: ', num2str(k, '%03d')]);
        	        kth_train_img_id = strrep(jth_train_il{k}, '.mat', '.png');

               		kth_train_img = imread([TRAIN_IMG_PATH,...
				'/', strrep(kth_train_img_id, REQ_PATHS.RM_STRING, '')]);
			kth_train_org_img = imread([TRAIN_IMG_ORG_PATH,...
                                '/', strrep(kth_train_img_id, 'edges', 'images')]);
			kth_train_org_img = imresize(kth_train_org_img, [96,96]);
			
			if(size(kth_train_img,3)~=3)
				kth_train_img = cat(3, kth_train_img,...
						       kth_train_img,...
						       kth_train_img);
			end

			if(size(kth_train_org_img,3)~=3)
				kth_train_org_img = cat(3, kth_train_org_img,...
							   kth_train_org_img,...
							   kth_train_org_img);
			end

			kth_train_img_l = double(rgb2ycbcr(kth_train_img));
			kth_train_org_img_l = double(rgb2ycbcr(kth_train_org_img));
			nn_l_diff_data = (kth_train_org_img_l - ...
						kth_train_img_l);

			[q_f2] = compute_hypercolumn_feat(kth_train_img,...
							 mean_data, net);
			[q_x, q_y] = meshgrid(1:1:size(q_f2,1), 1:1:size(q_f2,1));
			q_y = q_y(:); q_x = q_x(:);
			q_xy = [q_x, q_y];
 
			% find the nearest neighbors --
			q_f2 = reshape(q_f2, [size(q_f2,1)*size(q_f2,2),size(q_f2,3)]);	
			q_f2_norm = repmat(sqrt(sum(q_f2.*q_f2,2)),...
                       	        		  1, size(q_f2,2));
			q_f2 = q_f2./(q_f2_norm+eps);
			q_l_data = reshape(nn_l_diff_data, ...
					[size(nn_l_diff_data,1)*size(nn_l_diff_data,2),3]);

			% get the score for pixels --
			% now this would depend on the surroundings that we are interested in --
			if(k_pix_patch == 1)
				k_pix_score = sum(f2_sel.*q_f2,2);
	                        I = 1:length(k_pix_score);	
				Y = k_pix_score;	
			elseif(k_pix_patch > 1)
				k_pix_score = f2_sel*q_f2';
	                	k_pix_score = k_pix_score.*img_pat_loc;
		                [Y,I] = max(k_pix_score, [], 2);
			elseif(k_pix_patch == -1)
				k_pix_score = f2_sel*q_f2';
                                [Y,I] = max(k_pix_score, [], 2);
			else
				error('not recognized format');
			end		
	
			% find where Y is greater than best pixel score --
        	        k_places = Y > nn_pixel_scor;
             		nn_pixel_scor(k_places) = Y(k_places);
			nn_pixel_loxy(k_places,:) = q_xy(I(k_places),:);
			nn_pixel_NNid(k_places) = k;
			%nn_pixel_ldata(k_places) = q_l_data(I(k_places));
			nn_pixel_ldata = nn_pixel_ldata + (repmat(Y,1,3).*q_l_data);
			nn_pixel_lscore = nn_pixel_lscore + Y;

	                % find where Y is greater than best pixel score --
        	        %j_places = Y > nn_pixel_scor;
                	%nn_pixel_scor(j_places) = Y(j_places);
               		%nn_pixel_loxy(j_places,:) = q_xy(I(j_places),:);
                	%nn_pixel_NNid(j_places) = j;
                	%nn_pixel_ldata = nn_pixel_ldata + ith_det(j)*(Y.*q_l_data);
		end

		% using the nn-images and correspondences
		% create the hi-res version of the output of
		% image super-resolution algorithm --
		im_ycbcr = double(rgb2ycbcr(im));
		nn_pixel_ldata = nn_pixel_ldata./repmat(nn_pixel_lscore,1,3);
		nn_pixel_ldata = reshape(nn_pixel_ldata, [size(im_ycbcr,1), size(im_ycbcr,2),3]);
		im_ycbcr = im_ycbcr + nn_pixel_ldata;
		im_hr = ycbcr2rgb(uint8(im_ycbcr));	

		imwrite(im_hr, [DUMP_DATA_PATH,  '/' ith_image_name, '/',....
 		       	           ith_image_name, '_', num2str(j, '%03d') '.png']);
		%unlock([DUMP_DATA_PATH,'/', ith_image_name, '/', ith_image_name]);
	end
	unlock([DUMP_DATA_PATH,'/', ith_image_name, '/', ith_image_name]);
end

caffe.reset_all;
end

function [f2] = compute_hypercolumn_feat(im, mean_data, net)

        if(size(im,3)~=3)
                im = cat(3, im, im, im);
        end
        im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
        im_data = permute(im_data, [2, 1, 3]);  % flip width and height
        im_data = single(im_data);  % convert from uint8 to single
        im_data = im_data - mean_data;
        ims(:,:,:,1) = im_data;


        crop_height=96; crop_width = 96;
        net.blobs('data').reshape([crop_height+2*64, crop_width+2*64, 3, 1]);
        net.blobs('pixels').reshape([3,crop_height*crop_width]);

        h = crop_height;
        w = crop_width;
        hw = h * w;

        xs = reshape(repmat(0:w-1,h,1), 1, hw) + 64;
        ys = reshape(repmat(0:h-1,w,1)', 1, hw)+ 64;

        input_data = zeros(96+2*64,96+2*64,3,1);
        input_data(64+1:w+64, 64+1:h+64, :, 1) = ims;
        net.blobs('data').set_data(input_data);

        input_index = zeros(3, 96*96);
        input_index(1,:) = 0;
        input_index(2,:) = xs;
        input_index(3,:) = ys;
        net.blobs('pixels').set_data(input_index);

        % --
        net.forward_prefilled();
        out = net.blobs('fc5').get_data();
        f2 = out';
        f2 = reshape(f2, [96, 96,size(out,1)]);
        f2 = permute(f2, [2,1,3]);
end
