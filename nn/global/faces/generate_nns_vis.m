% generate the visualization of nns 
function generate_nns_vis()

% read the list of query images ---
CACHENAME = 'faces-edges';

VAL_DATA_PATH = ['./cachedir/', CACHENAME, '/feats/val/'];
VAL_IMG_PATH = ['./cachedir/faces-edges-val/']; 
val_img_list = dir([VAL_DATA_PATH, '*.mat']);
val_img_list = {val_img_list.name};
val_img_list = val_img_list';

CACHE_DIR = ['./cachedir/', CACHENAME,'/nns_conv5/'];

TRAIN_DATA_PATH = ['./cachedir/', CACHENAME, '/feats/train/'];
TRAIN_IMG_PATH = ['./cachedir/faces-edges/train/'];
train_img_list = dir([TRAIN_DATA_PATH, '*.mat']);
train_img_list = {train_img_list.name};
train_img_list = train_img_list';

NN_VIS = 5;
DUMP_DATA_PATH = [CACHE_DIR, 'vis/'];
if(~isdir(DUMP_DATA_PATH))
	mkdir(DUMP_DATA_PATH);
end

% initialize the index file --
indexPath = [DUMP_DATA_PATH, 'index.html'];
indexFile = fopen(indexPath, 'w');
writeHead(indexFile);

% for each query img, 
for i = 1:length(val_img_list)

        display(['Query Img: ', val_img_list{i}]);
	
	ith_img_id = val_img_list{i};
	ith_val_img = imread([VAL_IMG_PATH,...
			strrep(ith_img_id, '.mat', '.png')]);

	ith_val_data = load([CACHE_DIR, ith_img_id], ...
			'det_scores', 'img_list');

	ith_det = ith_val_data.det_scores;
	ith_train_il = ith_val_data.img_list;


	[~,I] = sort(ith_det, 'descend');
	I = I(1:NN_VIS);
	ith_det = ith_det(I);
	ith_train_il = ith_train_il(I);
	
        % initialiaze paths
        writeFirstRow(indexFile, i, NN_VIS+1);
        ith_data_path = sprintf('%s/%s',...
		DUMP_DATA_PATH, strrep(val_img_list{i},'.mat',''));
        ith_data_path_w = sprintf('./%s',strrep(val_img_list{i},'.mat',''));
        if(~isdir(ith_data_path))
                mkdir(ith_data_path);
        end

        % print the image --
	imgID  = strrep(ith_img_id, '.mat', '');
	ith_img_rs_det_path = sprintf('%s/%s.jpg',...
                                        ith_data_path, imgID);
        ith_img_rs_det_path_w = sprintf('%s/%s.jpg',...
                                         ith_data_path_w, imgID);
        ith_img_org_det_path = sprintf('%s/%s.jpg',...
                                       ith_data_path, imgID);
        ith_img_org_det_path_w = sprintf('%s/%s.jpg',...
                                       ith_data_path_w, imgID);

        % dump the images in cache --
        imwrite(ith_val_img, ith_img_rs_det_path);
        imwrite(ith_val_img, ith_img_org_det_path);
        writeImageInput(indexFile, ith_img_org_det_path_w,...
                        ith_img_rs_det_path_w, 'Input');
	
	% dump the results of NN --
	for j = 1:NN_VIS
	
		% read the required image --
		jth_train_img_id = strrep(ith_train_il{j}, '.mat', '.png');
		jth_train_img = imread([TRAIN_IMG_PATH,...
					'/', jth_train_img_id]);

		% get the required paths
        	ith_tr_rs_det_path =  sprintf('%s/%s',...
                                           ith_data_path, jth_train_img_id);
        	ith_tr_rs_det_path_w = sprintf('%s/%s',...
                                           ith_data_path_w, jth_train_img_id);
        	ith_tr_org_det_path = sprintf('%s/%s',...
                                           ith_data_path, jth_train_img_id);
        	ith_tr_org_det_path_w = sprintf('%s/%s',...
                                           ith_data_path_w, jth_train_img_id);
        	% dump the images in cache --
        	imwrite(jth_train_img, ith_tr_rs_det_path);
       	 	imwrite(jth_train_img, ith_tr_org_det_path);
        	writeImageInput(indexFile, ith_tr_org_det_path_w,...
                	        ith_tr_rs_det_path_w, ....
				['NN :', num2str(j, '%02d'), ' Score: ', num2str(ith_det(j))]);

	end

        %%
        % close the loop
        fprintf(indexFile,'</tr>\n<tr>\n');
        fprintf(indexFile,'</tr>\n</table></center>\n\n');

end

fprintf(indexFile,'</body>\n</html>\n');
fclose(indexFile);

end
function writeFirstRow(file, i,  n)

        fprintf(file, ['<h2>',int2str(i),'.', ' </h2>\n']);
        fprintf(file,'<center><table border="1">\n');
        fprintf(file,'<COLGROUP>\n');
        for i=1:n
           fprintf(file,'<COL width="100">');
        end
        fprintf(file,'\n<THEAD>\n<tr>\n');
end

function writeImageInput(file,im,tim,text)
        fprintf(file, ['<td><center><b>', text, '<b></center><a href="'...
                im,'" title="'...
                text,'" class="thickbox"><img src="'...
                tim,'" alt="Sorry, No Display!" border="0"/></a></td>\n']);
end

function writeHead(file)

        fprintf(file, '<html><head><title>Detection Results</title>\n');
        fprintf(file, '<script language="javascript" type="text/javascript" src="javascript/jquery.js"></script>\n');
        fprintf(file, '<link rel="stylesheet" href="css/thickbox.css" type="text/css" media="screen" /></head>\n');
        fprintf(file, '<body>\n');
        fprintf(file, '<script language="javascript" type="text/javascript" src="javascript/thickbox.js"></script>\n');
        fprintf(file, '<center><h1>Detection Results</h1></center>\n');
end
