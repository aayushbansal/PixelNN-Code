% we get the correspondences using the SegNet(Linear)
% and copy the missing data from nearest neighbors --

function copy_data_normals()

	% read the list of query images ---
	CACHENAME = 'catsanddogs-normals';

	VAL_DATA_PATH = ['./cachedir/', CACHENAME, '/feats/val/'];
	VAL_IMG_PATH = ['./cachedir/catsanddogs-normals/val/'];
	val_img_list = dir([VAL_DATA_PATH, '*.mat']);
	val_img_list = {val_img_list.name};
	val_img_list = val_img_list';

	CACHE_DATA_PATH = ['./cachedir/', CACHENAME,'/nns_conv5/'];

	TRAIN_DATA_PATH = ['./cachedir/', CACHENAME, '/feats/train/'];
	TRAIN_IMG_PATH = ['./cachedir/catsanddogs-normals/train/'];
	TRAIN_IMG_ORG_PATH = ['./cachedir/catsanddogs-normals/train/'];
	train_img_list = dir([TRAIN_DATA_PATH, '*.mat']);
	train_img_list = {train_img_list.name};
	train_img_list = train_img_list';
	
	% --
	NN_VIS = 1;
	DUMP_DATA_PATH = ['./cachedir/', CACHENAME, '/hi-res-NN1_org/'];
	if(~isdir(DUMP_DATA_PATH))
        	mkdir(DUMP_DATA_PATH);
	end

	% initialize the network --
	% set gpu --
	caffe.set_mode_gpu();
	gpu_id = 0;
	caffe.set_device(gpu_id);

	% load neural network in memory --
	model_dir = './data/net/vgg16_seg_hypercol/';
	net_model = [model_dir 'deploy.prototxt'];
	net_weights = [model_dir 'vgg16_seg_hypercol.caffemodel'];
	phase = 'test';

	mean_data = load([model_dir, 'mean_file.mat'],'mean_data');
	mean_data = mean_data.mean_data;
	mean_data = imresize(mean_data, [96,96], 'nearest');

	% Initialize a network
	net = caffe.Net(net_model, net_weights, phase);
	%[img_pat_loc] = get_img_pat_loc(im, 2);

	% for each query img --
	for i = 1:length(val_img_list)

        	ith_image_name = strrep(val_img_list{i}, '.mat', '');
        	display(['Image ID: ', ith_image_name]);

        	if(exist([DUMP_DATA_PATH,  '/',....
                	   ith_image_name, '.png'], 'file'))
                	continue;
       		end

		% -- 
        	% read the image --
        	im = imread([VAL_IMG_PATH, ith_image_name,'.png']);
		if(i == 1)
			[img_pat_loc] = get_img_pat_loc(im, 2);
		end
		%im_org = imread([TRAIN_IMG_ORG_PATH,ith_image_name, '.png']);

		if(size(im,3)~=3)
			im = cat(3, im, im, im);
		end

		% --
		[f2_sel] = compute_hypercolumn_feat(im, mean_data, net);
		[x_sft, y_sft] = meshgrid(1:1:size(f2_sel,1), 1:1:size(f2_sel,1));
         	y_sft = y_sft(:); x_sft = x_sft(:);

		% normalize the data --	
		f2_sel_norm = repmat(sqrt(sum(f2_sel.*f2_sel,2)),...
                	                 1, size(f2_sel,2));	
		f2_sel = f2_sel./(f2_sel_norm+eps);
		f2_sel = reshape(f2_sel, [size(f2_sel,1)*size(f2_sel,2),size(f2_sel,3)]);

		%[img_pat_loc] = get_img_pat_loc(im, 5);

		% -- 
		% get the nearest neighbor data --
       	 	ith_val_data = load([CACHE_DATA_PATH, ith_image_name, '.mat'], ...
                        'det_scores', 'img_list');
        	ith_det = ith_val_data.det_scores;
        	ith_train_il = ith_val_data.img_list;

        	[~,I] = sort(ith_det, 'descend');
        	I = I(1:NN_VIS);
        	ith_det = ith_det(I);
       		ith_train_il = ith_train_il(I);

		% dump the NN-train hypercol data
		nn_pixel_scor = zeros(size(f2_sel,1),1);
        	nn_pixel_loxy = zeros(size(f2_sel,1),2);
		nn_pixel_NNid = zeros(size(f2_sel,1),1);
		nn_pixel_ldata = zeros(size(f2_sel,1),1);
		nn_pixel_ldata_norm = zeros(size(f2_sel,1),1);	

		nn_org_img_data = cell(NN_VIS,1);
		nn_img_data = cell(NN_VIS,1);
		nn_l_diff_data = cell(NN_VIS,1);

		for j = 1:NN_VIS

                	% read the required image --
			display(['Nearest Neighbor: ', num2str(j, '%03d')]);
                	jth_train_img_id = strrep(ith_train_il{j}, '.mat', '.png');
			
                	jth_train_img = imread([TRAIN_IMG_PATH,	'/', jth_train_img_id]);
			jth_train_org_img = imread([TRAIN_IMG_ORG_PATH,...
                        	        '/', strrep(jth_train_img_id,'normals', 'dataset-catsanddog')]);
			%jth_train_org_img = imresize(jth_train_org_img,...
			%				 [size(jth_train_img,1),...
			%				 size(jth_train_img,2)]);
	
			if(size(jth_train_img,3)~=3)
				jth_train_img = cat(3, jth_train_img,...
						       jth_train_img,...
						       jth_train_img);
			end

			if(size(jth_train_org_img,3)~=3)
				jth_train_org_img = cat(3, jth_train_org_img,...
							   jth_train_org_img,...
							   jth_train_org_img);
			end

			jth_train_img_l = double(rgb2ycbcr(jth_train_img));
			jth_train_org_img_l = double(rgb2ycbcr(jth_train_org_img));
			nn_l_diff_data{j} = (jth_train_org_img_l(:,:,1) - ...
						jth_train_img_l(:,:,1));

			nn_org_img_data{j} = jth_train_org_img;
			nn_img_data{j} = jth_train_img;

			[q_f2] = compute_hypercolumn_feat(jth_train_img,...
							 mean_data, net);
			[q_x, q_y] = meshgrid(1:1:size(q_f2,1), 1:1:size(q_f2,1));
			q_y = q_y(:); q_x = q_x(:);
			q_xy = [q_x, q_y];
 
			%save([DUMP_DATA_PATH, ith_image_name, '/nn_',....
                	%       num2str(j, '%03d'),'_', jth_train_img_id, '.mat'], 'f2');
			% find the nearest neighbors --
			q_f2 = reshape(q_f2, [size(q_f2,1)*size(q_f2,2),size(q_f2,3)]);	
			q_f2_norm = repmat(sqrt(sum(q_f2.*q_f2,2)),...
                        	       		  1, size(q_f2,2));
			q_f2 = q_f2./(q_f2_norm+eps);
			q_l_data = reshape(nn_l_diff_data{j}, ...
					[size(nn_l_diff_data{j},1)*size(nn_l_diff_data{j},2),1]);

			% get the score for pixels --
			%j_pix_score = sum(f2_sel.*q_f2,2);
			%j_pix_score = f2_sel*q_f2';
			%j_pix_score = j_pix_score.*img_pat_loc;
                	%[Y,I] = max(j_pix_score, [], 2);
			%I = 1:length(j_pix_score);
			%[Y,I] = max(j_pix_score, [], 2);
			j_pix_score = sum(f2_sel.*q_f2,2);
			I = 1:length(j_pix_score);
			
			% find where Y is greater than best pixel score --
                	j_places = j_pix_score > nn_pixel_scor;
        	        nn_pixel_scor(j_places) = j_pix_score(j_places);
			nn_pixel_loxy(j_places,:) = q_xy(I(j_places),:);
			nn_pixel_NNid(j_places) = j;
			nn_pixel_ldata(j_places) = q_l_data(I(j_places));

	                % find where Y is greater than best pixel score --
        	        %j_places = Y > nn_pixel_scor;
               		%nn_pixel_scor(j_places) = Y(j_places);
	                %nn_pixel_loxy(j_places,:) = q_xy(I(j_places),:);
        	        %nn_pixel_NNid(j_places) = j;
 	                %nn_pixel_ldata(j_places) = q_l_data(I(j_places));	
			% there seems to be something wrong here --
			% needs correction --
			%nn_pixel_ldata = nn_pixel_ldata + ith_det(j)*(Y.*q_l_data);
			%nn_pixel_ldata_norm = nn_pixel_ldata_norm + ith_det(j);
		
		end

		% using the nn-images and correspondences
		% create the hi-res version of the output of
		% image super-resolution algorithm --
		im_ycbcr = double(rgb2ycbcr(im));
		nn_pixel_ldata = reshape(nn_pixel_ldata, [size(im_ycbcr,1), size(im_ycbcr,2)]);
		im_ycbcr(:,:,1) = im_ycbcr(:,:,1) + nn_pixel_ldata;
		im_hr = ycbcr2rgb(uint8(im_ycbcr));	

		imwrite(im_hr, [DUMP_DATA_PATH,  '/',....
        	           ith_image_name, '.png']);
	
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

function [img_pat_loc] = get_img_pat_loc(im, pat)

        crop_height = size(im,2);
        img_pat_loc = zeros(crop_height*crop_height, crop_height*crop_height);
        iter = 1;
        for j = 1:crop_height
                for i = 1:crop_height

                        img_pat = zeros(crop_height, crop_height);
                        img_pat(i,j) = 1;

                        st_pos_i = min(max(i - pat, 1), crop_height);
                        st_pos_j = min(max(j - pat, 1), crop_height);
                        end_pos_i = min(max(st_pos_i+2*pat,1), crop_height);
                        end_pos_j = min(max(st_pos_j+2*pat,1), crop_height);

                        img_pat(st_pos_i:end_pos_i, st_pos_j:end_pos_j) = 1;
                        img_pat_loc(iter,:) = img_pat(:);
                        iter = iter+1;
                end
        end
end
