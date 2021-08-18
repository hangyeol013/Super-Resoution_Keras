
import warnings
warnings.filterwarnings(action = 'ignore')

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


import Basic_model_MS_LapSRN as Basic_model
from utils import psnr
from utils import ssim
from utils import adjust_learning_rate
import prepare_sub_image_Three_inputs as ps
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import cv2
import numpy as np
from scipy import io
import keras.backend as K
from collections import defaultdict
import time
import pandas as pd


# ## Test

# In[5]:


scale = 2
test_set = 'Urban100'
conv_side = 8
alpha = 0.5
D = 5
R = 8


def test():
    
    model4 = Basic_model.MS_LapSRN_model(D = D, R = R)
    
    model4.compile(optimizer = 'adam', loss = 'mean_absolute_error')
    
    model4.load_weights('checkpoint/MS_LapSRN_DIV2K_alpha:{}_2.h5'.format(alpha))
    
    print(model4.summary())

    PATH_image = '../../Dataset/MS_LapSRN/Test/{}/'.format(test_set)
    IMAGE_PATH_x2 = 'output_image/{}/scale{}'.format(test_set, scale)
    IMAGE_PATH_x4 = 'output_image/{}/scale{}'.format(test_set, scale*2)
    
    names_image = os.listdir(PATH_image)
    names_image = sorted(names_image)
    
    nums = len(names_image)
    
    count = 0
    global total_history
    
    
    for i in range(0, 1):

        psnr_model4_x2 = []
        psnr_bicubic_x2 = []
        
        psnr_model4_x4 = []
        psnr_bicubic_x4 = []
        
        ssim_model4_x2 = []
        ssim_bicubic_x2 = []
        
        ssim_model4_x4 = []
        ssim_bicubic_x4 = []
        
        total_time = []
        
        
        for i in range(nums):
            
            
            OUTPUT_NAME_x2 = IMAGE_PATH_x2 + '/' + 'MS_LapSRN_alpha{}_x2_{}.png'.format(alpha, i)
            OUTPUT_NAME_x4 = IMAGE_PATH_x4 + '/' + 'MS_LapSRN_alpha{}_x4_{}.png'.format(alpha, i)

            mat_image = io.loadmat(PATH_image + names_image[i])
            
            input_img = mat_image['im_input_rgb']
            
            hr_img_x2 = mat_image['im_hr_x2_rgb']
            bicubic_img_x2 = mat_image['im_bicubic_x2_rgb']
            
            hr_img_x4 = mat_image['im_hr_x4_rgb']
            bicubic_img_x4 = mat_image['im_bicubic_x4_rgb']
            
            shape_input = input_img.shape
            shape_x2 = hr_img_x2.shape
            shape_x4 = hr_img_x4.shape
            
            input_RGB = np.zeros([1, shape_input[0], shape_input[1], 3])
            input_RGB[0, :, :, :] = input_img / 255
            
            start_time = time.time()

            pre = model4.predict(input_RGB, batch_size = 1)
            pre_x2 = pre[0]
            pre_x4 = pre[1]
            
            pre_x2 = pre_x2 * 255
            pre_x4 = pre_x4 * 255
            
            pre_x2[pre_x2[:] > 255] = 255
            pre_x2[pre_x2[:] < 0] = 0
            pre_x4[pre_x4[:] > 255] = 255
            pre_x4[pre_x4[:] < 0] = 0
            
            finish_time = time.time()
            
            output_img_x2 = np.zeros([shape_x2[0], shape_x2[1], 3])
            output_img_x2[:, :, 2] = pre_x2[0, :, :, 0]
            output_img_x2[:, :, 1] = pre_x2[0, :, :, 1]
            output_img_x2[:, :, 0] = pre_x2[0, :, :, 2]
            
            cv2.imwrite(OUTPUT_NAME_x2, output_img_x2)
            
            hr_img_x2_r = hr_img_x2[:, :, 0]
            hr_img_x2_g = hr_img_x2[:, :, 1]
            hr_img_x2_b = hr_img_x2[:, :, 2]
            
            output_img_x2_r = output_img_x2[:, :, 2]
            output_img_x2_g = output_img_x2[:, :, 1]
            output_img_x2_b = output_img_x2[:, :, 0]
            
            bicubic_img_x2_r = bicubic_img_x2[:, :, 0]
            bicubic_img_x2_g = bicubic_img_x2[:, :, 1]
            bicubic_img_x2_b = bicubic_img_x2[:, :, 2]
            
            hr_img_x2_Y = 16 + (65.738 * hr_img_x2_r + 129.057 * hr_img_x2_g + 25.064 * hr_img_x2_b) / 255
            output_img_x2_Y = 16 + (65.738 * output_img_x2_r + 129.057 * output_img_x2_g + 25.064 * output_img_x2_b) / 255
            bicubic_img_x2_Y = 16 + (65.738 * bicubic_img_x2_r + 129.057 * bicubic_img_x2_g + 25.064 * bicubic_img_x2_b) / 255
            
            
            output_img_x4 = np.zeros([shape_x4[0], shape_x4[1], 3])
            output_img_x4[:, :, 2] = pre_x4[0, :, :, 0]
            output_img_x4[:, :, 1] = pre_x4[0, :, :, 1]
            output_img_x4[:, :, 0] = pre_x4[0, :, :, 2]
            
            cv2.imwrite(OUTPUT_NAME_x4, output_img_x4)
            
            hr_img_x4_r = hr_img_x4[:, :, 0]
            hr_img_x4_g = hr_img_x4[:, :, 1]
            hr_img_x4_b = hr_img_x4[:, :, 2]
            
            output_img_x4_r = output_img_x4[:, :, 2]
            output_img_x4_g = output_img_x4[:, :, 1]
            output_img_x4_b = output_img_x4[:, :, 0]
            
            bicubic_img_x4_r = bicubic_img_x4[:, :, 0]
            bicubic_img_x4_g = bicubic_img_x4[:, :, 1]
            bicubic_img_x4_b = bicubic_img_x4[:, :, 2]
            
            hr_img_x4_Y = 16 + (65.738 * hr_img_x4_r + 129.057 * hr_img_x4_g + 25.064 * hr_img_x4_b) / 255
            output_img_x4_Y = 16 + (65.738 * output_img_x4_r + 129.057 * output_img_x4_g + 25.064 * output_img_x4_b) / 255 
            bicubic_img_x4_Y = 16 + (65.738 * bicubic_img_x4_r + 129.057 * bicubic_img_x4_g + 25.064 * bicubic_img_x4_b) / 255
            
            
            # YCrCb Channel에서 Y에 대해 PSNR 측정 시
            hr_img_x2_measure = hr_img_x2_Y[conv_side:-conv_side, conv_side:-conv_side]
            output_img_x2_measure = output_img_x2_Y[conv_side:-conv_side, conv_side:-conv_side]
            bicubic_img_x2_measure = bicubic_img_x2_Y[conv_side:-conv_side, conv_side:-conv_side]
            
            psnr_x2 = psnr(output_img_x2_measure, hr_img_x2_measure)
            ssim_x2 = ssim(output_img_x2_measure, hr_img_x2_measure)
            
            psnr_x2_bicubic = psnr(bicubic_img_x2_measure, hr_img_x2_measure)
            ssim_x2_bicubic = ssim(bicubic_img_x2_measure, hr_img_x2_measure)
            
            hr_img_x4_measure = hr_img_x4_Y[conv_side:-conv_side, conv_side:-conv_side]
            output_img_x4_measure = output_img_x4_Y[conv_side:-conv_side, conv_side:-conv_side]
            bicubic_img_x4_measure = bicubic_img_x4_Y[conv_side:-conv_side, conv_side:-conv_side]
            
            psnr_x4 = psnr(output_img_x4_measure, hr_img_x4_measure)
            ssim_x4 = ssim(output_img_x4_measure, hr_img_x4_measure)
            
            psnr_x4_bicubic = psnr(bicubic_img_x4_measure, hr_img_x4_measure)
            ssim_x4_bicubic = ssim(bicubic_img_x4_measure, hr_img_x4_measure)
            
            print(i + 1)
            
            print('Bicubic_x2: ', psnr_x2_bicubic, 'ssim: ', ssim_x2_bicubic)
            print('Model4_x2: ', psnr_x2, 'ssim: ', ssim_x2)
            
            print('Bicubic_x4: ', psnr_x4_bicubic, 'ssim: ', ssim_x4_bicubic)
            print('Model4_x4: ', psnr_x4, 'ssim: ', ssim_x4)
            
            time_network = finish_time - start_time
            print('time: ', time_network)
                        
            
            psnr_bicubic_x2.append(psnr_x2_bicubic)
            ssim_bicubic_x2.append(ssim_x2_bicubic)
            
            psnr_model4_x2.append(psnr_x2)
            ssim_model4_x2.append(ssim_x2)
            
            psnr_bicubic_x4.append(psnr_x4_bicubic)
            ssim_bicubic_x4.append(ssim_x4_bicubic)
            
            psnr_model4_x4.append(psnr_x4)
            ssim_model4_x4.append(ssim_x4)
            
            total_time.append(time_network)
            
            
        model_x2_dataframe = pd.DataFrame(psnr_model4_x2)
        model_x4_dataframe = pd.DataFrame(psnr_model4_x4)

        model_x4_dataframe.to_csv('excel/{}_x4_D{}R{}_alpha:{}.csv'.format(test_set, D, R, alpha), header = ['model_alpha:{}'.format(alpha)], index = True)
        
        psnr_bicubic_x2_final = np.mean(psnr_bicubic_x2)
        ssim_bicubic_x2_final = np.mean(ssim_bicubic_x2)
        
        psnr_model4_x2_final = np.mean(psnr_model4_x2)
        ssim_model4_x2_final = np.mean(ssim_model4_x2)
        
        
        psnr_bicubic_x4_final = np.mean(psnr_bicubic_x4)
        ssim_bicubic_x4_final = np.mean(ssim_bicubic_x4)
        
        psnr_model4_x4_final = np.mean(psnr_model4_x4)
        ssim_model4_x4_final = np.mean(ssim_model4_x4)
        
        time_final = np.sum(total_time)
        
        print('-------------------------------')
        print('Time: ', time_final)
        print('------------------------------')
        
        
        print('Bicubic_x2')
        print('PSNR: ', psnr_bicubic_x2_final, 'SSIM: ', ssim_bicubic_x2_final)
        print('Model4_x2')
        print('PSNR: ', psnr_model4_x2_final, 'SSIM: ', ssim_model4_x2_final)
        
        print('Bicubic_x4')
        print('PSNR: ', psnr_bicubic_x4_final, 'SSIM: ', ssim_bicubic_x4_final)
        print('Model4_x4')
        print('PSNR: ', psnr_model4_x4_final, 'SSIM: ', ssim_model4_x4_final)
        
        tf.keras.backend.clear_session()
        
    
if __name__ == '__main__':
    
    test()
