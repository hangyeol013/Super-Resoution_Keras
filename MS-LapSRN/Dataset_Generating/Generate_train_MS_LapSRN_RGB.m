clear;close all;
%% settings
%% ÇÒ¶§¸¶´Ù folder À§Ä¡ test¶û train ¹Ù²ãÁà¾ß ÇÑ´Ù.

folder = 'Train/DIV2K_train_HR/HR';
%folder = 'Test/DIV2K_validation/HR';

savepath_train = 'train_DIV2K_scale4_RGB.h5';
savepath_test = 'val_DIV2K_scale4_RGB.h5';

size_input = 32;
size_label_x2 = 64;
size_label_x4 = 128;

scale = 4;
stride = 256;


%% initialization
data = zeros(size_input, size_input, 3, 1);
label_x2 = zeros(size_label_x2, size_label_x2, 3, 1);
label_x4 = zeros(size_label_x4, size_label_x4, 3, 1);

count = 0;


%% generate data

filepaths = dir(fullfile(folder,'*.png'));
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = im2double(image);
    im_label_x4 = modcrop(image, scale);
    [hei,wid,rgb] = size(im_label_x4);

    for y = 1 : stride : hei-size_label_x4+1
        for x = 1 :stride : wid-size_label_x4+1
            
            subim_label_x4 = im_label_x4(y : y+size_label_x4-1, x : x+size_label_x4-1, :);
            subim_label_x2 = imresize(subim_label_x4, 1/scale*2, 'bicubic');
            subim_input = imresize(subim_label_x4, 1/scale, 'bicubic');

            count=count+1;
            data(:, :, :, count) = subim_input;
            label_x2(:, :, :, count) = subim_label_x2;
            label_x4(:, :, :, count) = subim_label_x4;
        end
    end
end

data_dims = size(data);
label_dims_x2 = size(label_x2);
label_dims_x4 = size(label_x4);


%% writing to HDF5

h5create(savepath_train,'/data',data_dims);
h5create(savepath_train,'/label_x2', label_dims_x2);
h5create(savepath_train, '/label_x4', label_dims_x4);

h5write(savepath_train, '/data', data);
h5write(savepath_train, '/label_x2', label_x2);
h5write(savepath_train, '/label_x4', label_x4);

h5disp(savepath_train);