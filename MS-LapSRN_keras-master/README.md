# MS-LapSRN_keras


A implementation of the original paper ['Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks'](https://arxiv.org/abs/1710.01992)



<center><img width = "800" src="https://user-images.githubusercontent.com/58276840/107219462-6798fd80-6a11-11eb-91f1-6d9c7374e9d8.png"></center>


tensorflow-gpu 2.0.0, keras 2.3.1 based implementation on Python 3.6.9, using Jupyter Notebook.




### Implementation
-------------------------------------------------------
My implementation may have some differences with the original paper:


#### Networks)

- Input image size: 3 (RGB)
- kernel size: 3x3 (Upsampling layer: 4x4)
- Zero-padding for all layers
- filters: 64
- D (# of Distinct convolutional layers): 5
- R (# of Recursive blocks): 8
- Upsampling: Transposed Convolutional layers (Conv2DTrnapose)
- Activation Function: Leaky ReLU (alpha: 0.2)


#### Training)

- Loss Function: MAE (Mean Absolute Error, L1)
- Optimizer: Adam
- Learning rate: 10e-4 (Decreased by half at every 100 epochs) 
- Batch size: 16


#### Dataset)

##### Training)
- DIV2K (800 images) (Training set)
- Other 20 images from the validation set of the BSD500 dataset (Validation set)
- Patch size: 48 (Input), 48 x scale (Label)


##### Test)
- Set5, Set14, BSD200, Urban100



### Use
-------------------------------------------------------

You can generate dataset (training sample, test sample) through matlab files in Dataset_Generating directory
- Excute for making LR images: `Generate_LR.m`
- Excute for training sample: `Generate_train_MS_LapSRN_RGB.m`
- Excute for test sample: `Generate_test_MS_LapSRN.m`


I also uploaded the trained weight files.

With Test.ipynb file and weight files in 'weights' directory, you can test the network.
