clear;close all;
%% settings
dataset = 'Set5';
folder = ['Test/', dataset, '/HR'];

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

scale = 2;

for i = 1 : length(filepaths)
    im_hr = imread(fullfile(folder,filepaths(i).name));
    im_hr_x4 = modcrop(im_hr, 4);
    [hr_hei, hr_wid, channel] = size(im_hr_x4);
%     
%     % ycbcr 데이터 만드는 부분
%     % Set14 흑백 이미지 고려
%     if size(im_hr_x4,3) > 1
%         im_hr_x4_ycbcr = rgb2ycbcr(im_hr_x4);
%     elseif size(im_hr_x4,3) == 1
%         img = zeros(hr_hei, hr_wid, 3);
%         im_hr_x4_ycbcr = rgb2ycbcr(img);
%         im_hr_x4_ycbcr(:, :, 1) = im_hr_x4;
%     end
%     
%     im_hr_x4_ycbcr = double(im_hr_x4_ycbcr);
%     
%     im_input_ycbcr = imresize(im_hr_x4, 1/(scale*2), 'bicubic');
%     im_hr_x2_ycbcr = imresize(im_hr_x4, 1/scale, 'bicubic');
%     
%     im_hr_x2_ycbcr = imresize(im_hr_x4_ycbcr, 1/scale, 'bicubic');
%     im_input_ycbcr = imresize(im_hr_x2_ycbcr, 1/scale, 'bicubic');
%     
    %rgb 데이터 만드는 부분
    im_hr_x4_rgb = double(im_hr_x4);
    
    %Set14 흑백 이미지 고려
    if size(im_hr_x4_rgb, 3) == 1
        im_hr_rgb = zeros(hr_hei, hr_wid, 3);
        im_hr_rgb(:, :, 1) = 1.164 * (im_hr_x4_rgb - 16);
        im_hr_rgb(:, :, 2) = 1.164 * (im_hr_x4_rgb - 16);
        im_hr_rgb(:, :, 3) = 1.164 * (im_hr_x4_rgb - 16);
        im_hr_x4_rgb = im_hr_rgb;
    end
    
    im_input_rgb = imresize(im_hr_x4_rgb, 1/(scale*2), 'bicubic');
    im_hr_x2_rgb = imresize(im_hr_x4_rgb, 1/scale, 'bicubic');
    
    %im_hr_x2_rgb = imresize(im_hr_x4_rgb, 1/scale, 'bicubic');
    %im_input_rgb = imresize(im_hr_x2_rgb, 1/scale, 'bicubic');
    
    im_bicubic_x4_rgb = imresize(im_input_rgb, scale*2, 'bicubic');
    im_bicubic_x2_rgb = imresize(im_input_rgb, scale, 'bicubic');

    filename = ['MS_LapSRN/Test/', dataset, '/', dataset, '_scale_', num2str(scale), '&', num2str(scale*2), '_', num2str(i), '.mat'];
    save(filename, 'im_input_rgb', 'im_hr_x2_rgb', 'im_hr_x4_rgb', 'im_bicubic_x2_rgb', 'im_bicubic_x4_rgb');
    
end