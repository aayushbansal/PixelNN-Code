function [im_data] = normals_image_provider(options, impath, flip)

% --
im = imread(impath);
if(flip)
	im = flipdim(im,2);
end

%
cnn_input_size = options.cnn_input_size;

% setting the im-data
im_data = single(im(:, :, [3, 2, 1]));  
im_data = permute(im_data, [2, 1, 3]);  
im_data = single(im_data);
% subtract mean_data (already in W x H x C, BGR)
im_data(:,:,1) = im_data(:,:,1) - options.meanvalue(1);  
im_data(:,:,2) = im_data(:,:,2) - options.meanvalue(2);
im_data(:,:,3) = im_data(:,:,3) - options.meanvalue(3);

end
