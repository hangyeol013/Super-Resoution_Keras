import tensorflow as tf
import numpy as np
from skimage import measure
import math


def ssim(target, ref):
    target_data = np.array(target)
    ref_data = np.array(ref)
    
    (score, diff) = measure.compare_ssim(target_data, ref_data, full = True, multichannel=True)
    
    return score

def psnr(target, ref):
    
    target_data = np.array(target, dtype = float)
    ref_data = np.array(ref, dtype = float)
    
    diff = ref_data - target_data
    diff = diff.flatten('C')
    
    rmse = math.sqrt(np.mean(diff**2))
    
    return 20 * math.log10(255/rmse)


def adjust_learning_rate(epoch):
    
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 100
    
    lr = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
    return lr

