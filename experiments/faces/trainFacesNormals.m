% this is an example code for training a PixelNet model using caffe
% here we consider a 224x224 image 
% uniform sampling of pixels in an image, used for surface normal estimation --
function trainFacesNormals(gpuid, options)

	method = 'faces-normals';
	cachepath = [options.cachepath, method, '/'];
	if(~isdir(cachepath))
		mkdir(cachepath);
	end

	% --
	caffe.reset_all;
	caffe.set_device(gpuid);
	caffe.set_mode_gpu;
        
	% train the network --
	trainNet(options, cachepath);
	caffe.reset_all;
end

function trainNet(options,  cachepath)

	% check if it has already been trained
	trainfolder = [cachepath, 'TRAIN/'];
	if(~isdir(trainfolder))
		mkdir(trainfolder);
	end


	% check if the dataset is ready
	% this would be available in the download_data.sh script
	datasetfile = ['./data/faces/normals.mat'];
	if ~(exist(datasetfile, 'file'))
    		error('Dataset is not prepared!');
	end
	load(datasetfile, 'imagelist', 'normalslist');
	%imagelist = strcat('./data/cityscape/', imagelist);
	%seglist = strcat('./data/cityscape/', seglist);
	imagelist = strcat('/mnt/pcie1/user/aayushb/Faces/', imagelist);
	normalslist = strcat('/mnt/pcie1/user/aayushb/Faces/', normalslist);

	% load the network --
	solverpath = ['./data/net/faces/solver.prototxt'];
	initmodelpath = ['./data/net/faces/VGG16_fconv.caffemodel'];
	trainModel(options,trainfolder,imagelist,normalslist,solverpath,initmodelpath);
end

function trainModel(options,trainfolder,imagelist,normalslist,solverpath,initmodelpath)


	% Though this is not required -- 
	rand_ids = randperm(length(imagelist));
	imagelist = imagelist(rand_ids);
	normalslist = normalslist(rand_ids);

	%
	li = length(imagelist);
	caffe.reset_all;

	% initialize network
	solver = caffe.get_solver(solverpath);
	% load the model
	solver.net.copy_from(initmodelpath);

	% --
	maxsize = options.cnn_input_size + 96;
	input_data = zeros(maxsize, maxsize, 3, options.normalsimbatch, 'single');
	input_index = zeros(3,options.normalsbatchsize, 'single');
	input_label = zeros(3,options.normalsbatchsize, 'single');
	solver.net.blobs('data').reshape([maxsize, maxsize, 3, options.normalsimbatch]);
	solver.net.blobs('pixels').reshape([3,options.normalsbatchsize]);
	solver.net.blobs('labels').reshape([3,options.normalsbatchsize]);

	% set up memory
	solver.net.forward_prefilled();
	% then start training
	oldrng = rng;
	rng(options.seed, 'twister');

	for epoch = 1:options.normalsepoch
   	 index = randperm(li);
   	 for i = 1:options.normalsimbatch:li
        	j = i+options.normalsimbatch-1;
        	if j > li
            	continue;
        	end

	        st = 1;
       		ed = options.normalssamplesize;
        	im = 0;
        	for k=i:j
            		ik = index(k);

            		if options.trainFlip
		                flip = rand(1) > 0.5;
            		end


	    		im_data = normals_image_provider(options, normalslist{ik}, flip);
	    		[sampled, label] = normals_label_provider(...
				options, imagelist{ik}, options.normalssamplesize, flip);

            		% notice the zero-index vs. the one-index
            		sampled(1,:) = im;
            		sampled(2,:) = sampled(2,:) + 48;
            		sampled(3,:) = sampled(3,:) + 48;

            		im = im + 1;
            		input_data(49:options.cnn_input_size+48,....
		      	 	   49:options.cnn_input_size+48, :, im) = im_data;
            		input_index(:, st:ed) = sampled;
            		input_label(:, st:ed) = label;
           		st = st + options.normalssamplesize;
            		ed = ed + options.normalssamplesize;
        	end

	        solver.net.blobs('data').set_data(input_data);
        	solver.net.blobs('pixels').set_data(input_index);
        	solver.net.blobs('labels').set_data(input_label);
        	solver.step(1);
        	% clean up everything
        	input_data(:) = 0;
    	end
    	% add another condition
    	if mod(epoch, options.saveEpoch) == 0 && epoch < options.normalsepoch
        	epochfile = [trainfolder, sprintf('(%02d).caffemodel',epoch)];
        	solver.net.save(epochfile);
    	end

	% -- 
      end

	targetfile = [trainfolder,'final_model.caffemodel'];
	solver.net.save(targetfile);
	caffe.reset_all;
	rng(oldrng);

end
