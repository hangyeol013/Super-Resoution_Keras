
import warnings
warnings.filterwarnings(action = 'ignore')

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


import Basic_model_MS_LapSRN as Basic_model
from utils import psnr
from utils import ssim
from utils import adjust_learning_rate
import prepare_sub_image_Three_inputs as ps
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import numpy as np
from scipy import io
import keras.backend as K
from collections import defaultdict
import matplotlib.pyplot as plt


# ## Train & Predict

scale = 2
test_set = 'Set5'
conv_side = 8
alpha = 0.5
D = 5
R = 8
total_history = defaultdict(list)


def train():
    
    model = Basic_model.MS_LapSRN_model(D = D, R = R)
    
    model.compile(optimizer = 'adam', loss = 'mean_absolute_error', loss_weights = {'add_8':1-alpha, 'add_9':alpha})
    print(model.summary())
    
    #model.load_weights('checkpoint/D{}R{}_DIV2K_alpha:{}.h5'.format(D, R, alpha))
    
    data, label_x2, label_x4 = ps.read_training_data('training_sample/train_DIV2K_scale4_RGB.h5')
    val_data, val_label_x2, val_label_x4 = ps.read_training_data('training_sample/val_DIV2K_scale4_RGB.h5')
    
    label = [label_x2, label_x4]
    val_label = [val_label_x2, val_label_x4]
    
    PATH_image = '../../Dataset/MS_LapSRN/Test/{}/'.format(test_set)
    
    names_image = os.listdir(PATH_image)
    names_image = sorted(names_image)
    
    nums = len(names_image)
    
    count = 0
    global total_history
    
    checkpoint_filepath = 'checkpoint/MS_LapSRN_DIV2K_alpha:{}_Wls.h5'.format(alpha)
    checkpoint_callbacks = [ModelCheckpoint(filepath = checkpoint_filepath, monitor = 'val_loss', verbose = 1, mode = 'min', 
                                            save_best_only = True), LearningRateScheduler(adjust_learning_rate)]
    
    for i in range(0, 2000):
        
        history = model.fit(x = data, y = label, batch_size = 16, epochs = 2, verbose = 1,
                            callbacks = checkpoint_callbacks, validation_data = (val_data, val_label), shuffle = True)
        
        count += 1
        
        psnr_model_x2 = []
        psnr_bicubic_x2 = []
        
        psnr_model_x4 = []
        psnr_bicubic_x4 = []
        
        ssim_model_x2 = []
        ssim_bicubic_x2 = []
        
        ssim_model_x4 = []
        ssim_bicubic_x4 = []
        
        
        for i in range(nums):
            
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

            pre = model.predict(input_RGB, batch_size = 1)
            pre_x2 = pre[0]
            pre_x4 = pre[1]
            
            pre_x2 = pre_x2 * 255
            pre_x4 = pre_x4 * 255
            
            pre_x2[pre_x2[:] > 255] = 255
            pre_x2[pre_x2[:] < 0] = 0
            pre_x4[pre_x4[:] > 255] = 255
            pre_x4[pre_x4[:] < 0] = 0
            

            output_img_x2 = np.zeros([shape_x2[0], shape_x2[1], 3])
            output_img_x2[:, :, 2] = pre_x2[0, :, :, 0]
            output_img_x2[:, :, 1] = pre_x2[0, :, :, 1]
            output_img_x2[:, :, 0] = pre_x2[0, :, :, 2]
            
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
            print('Model_x2: ', psnr_x2, 'ssim: ', ssim_x2)
            
            print('Bicubic_x4: ', psnr_x4_bicubic, 'ssim: ', ssim_x4_bicubic)
            print('Model_x4: ', psnr_x4, 'ssim: ', ssim_x4)

            
            psnr_bicubic_x2.append(psnr_x2_bicubic)
            ssim_bicubic_x2.append(ssim_x2_bicubic)
            
            psnr_model_x2.append(psnr_x2)
            ssim_model_x2.append(ssim_x2)
            
            psnr_bicubic_x4.append(psnr_x4_bicubic)
            ssim_bicubic_x4.append(ssim_x4_bicubic)
            
            psnr_model_x4.append(psnr_x4)
            ssim_model_x4.append(ssim_x4)
            
            
        psnr_bicubic_x2_final = np.mean(psnr_bicubic_x2)
        ssim_bicubic_x2_final = np.mean(ssim_bicubic_x2)
        
        psnr_model_x2_final = np.mean(psnr_model_x2)
        ssim_model_x2_final = np.mean(ssim_model_x2)
        
        
        psnr_bicubic_x4_final = np.mean(psnr_bicubic_x4)
        ssim_bicubic_x4_final = np.mean(ssim_bicubic_x4)
        
        psnr_model_x4_final = np.mean(psnr_model_x4)
        ssim_model_x4_final = np.mean(ssim_model_x4)
        
        print('Epochs: ', count*2)
        
        print('Bicubic_x2')
        print('PSNR: ', psnr_bicubic_x2_final, 'SSIM: ', ssim_bicubic_x2_final)
        print('Model_x2')
        print('PSNR: ', psnr_model_x2_final, 'SSIM: ', ssim_model_x2_final)
        
        print('Bicubic_x4')
        print('PSNR: ', psnr_bicubic_x4_final, 'SSIM: ', ssim_bicubic_x4_final)
        print('Model_x4')
        print('PSNR: ', psnr_model_x4_final, 'SSIM: ', ssim_model_x4_final)
        
                # Error Graph 그리기
        for key, value in history.history.items():

            total_history[key] = sum([total_history[key], history.history[key]], [])
            
        length = len(total_history['loss'])

        plt.plot(total_history['loss'])
        plt.plot(total_history['val_loss'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        #plt.xlim(9, length)
        plt.ylim(0.015, 0.003)
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        
    
if __name__ == '__main__':
    
    train()


