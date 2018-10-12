function [sampled, label] = normals_label_provider(options,  impath, sample_size, flip)

% first get the mask
try,

	cnn_input_size = options.cnn_input_size;
	bmask = single(imread(impath));

	if(flip)
		bmask = flipdim(bmask,2);
	end

	bmask = bmask/128 - 1;

	% then sample from each
	sampled = zeros(3, sample_size, 'single');
	label = zeros(3, sample_size, 'single');
	im_val = true(cnn_input_size,cnn_input_size);
	[y,x] = find(im_val == 1);

	% rand-ids -
	rand_ids = randperm(length(y));
	rand_ids = rand_ids(1:sample_size);
	y = y(rand_ids);
	x = x(rand_ids);

	sampled(2,:) = y - 1;
	sampled(3,:) = x - 1;

	% get image values at the pixels --
	for i = 1:sample_size
        	label(:,i) = bmask(y(i), x(i),:);
	end
catch,
	keyboard;
end


end

