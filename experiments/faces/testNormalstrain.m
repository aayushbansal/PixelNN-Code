% demo code to use the surface normal mode --
clc; clear all;

%
conv_cache = ['./cachedir/faces-normals/'];
if(~isdir(conv_cache))
        mkdir(conv_cache);
end

% initialize caffe
NET_FILE_PATH = ['./cachedir/faces-normals/TRAIN/'];
net_file     = [NET_FILE_PATH, '(02).caffemodel'];
DEPLOY_FILE_PATH = ['./data/net/faces/'];
deploy_file  = [DEPLOY_FILE_PATH, 'deploy.prototxt']; 

% set the gpu --
% if not using GPU, set it to CPU mode.
gpu_id = 0;
caffe.reset_all;
caffe.set_device(gpu_id);
caffe.set_mode_gpu;
net = caffe.Net(deploy_file, net_file, 'test');

cnn_input_size = 96;
crop_height = 96; crop_width = 96;
image_mean = cat(3,  102.9801*ones(cnn_input_size),...
		     115.9465*ones(cnn_input_size),...
		     122.7717*ones(cnn_input_size));

% read the image set for NYU
load(['./data/faces/normals.mat']);

% for each image in the img_set
for i = 1:length(normalslist)

	display(['Image : ', normalslist{i}]);
	ith_Img = im2uint8(imread(['/mnt/pcie1/user/aayushb/Faces/',...
					normalslist{i}]));
	ith_org_img = im2uint8(imread(['/mnt/pcie1/user/aayushb/Faces/',...
					imagelist{i}]));

	%
        save_file_name = [conv_cache, strrep(normalslist{i}, '/', '-')];
        if(exist(save_file_name, 'file'))
                continue;
        end

	save_file_org_name = [conv_cache, strrep(imagelist{i}, '/', '-')];
	 
        j_tmp = single(ith_Img(:,:,[3 2 1]));
        %j_tmp = imresize(j_ims, [cnn_input_size, cnn_input_size], ...
        %                   'bilinear', 'antialiasing', false);
        j_tmp = j_tmp - image_mean;
        ims(:,:,:,1) = permute(j_tmp, [2 1 3]);	

        %
        net.blobs('data').reshape([crop_height+96, crop_width+96, 3, 1]);
	net.blobs('pixels').reshape([3,crop_height*crop_width]);
        h = crop_height;
        w = crop_width;
        hw = h * w;

        xs = reshape(repmat(0:w-1,h,1), 1, hw) + 48;
        ys = reshape(repmat(0:h-1,w,1)', 1, hw)+ 48;


	% set the image data --
        input_data = zeros(crop_height+96,crop_width+96,3,1);
        input_data(48+1:crop_width+48, 48+1:crop_width+48, :, 1) = ims;
        net.blobs('data').set_data(input_data);
	
	% set the pixels --
        input_index = zeros(3, crop_height*crop_width);
        input_index(1,:) = 0;
        input_index(2,:) = xs;
        input_index(3,:) = ys;
        net.blobs('pixels').set_data(input_index);

	% feed forward the values --
        net.forward_prefilled();
        out = net.blobs('fc8_new').get_data();

        % reshape the data --
        f2 = out';
        f2 = reshape(f2, [crop_height, crop_width,3]);
        f2 = permute(f2, [2,1,3]);

        % back to 0-255
	f2 = uint8(128*(f2+1));
        imwrite(f2, strrep(save_file_name, '.jpg', '.png'));
	imwrite(ith_org_img, strrep(save_file_org_name, '.jpg', '.png'));

end

% reset caffe
caffe.reset_all;
