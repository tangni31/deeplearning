# Codes for T15's project Image Super-resolution
----


## Dataset: [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)   

Note: You should download `Align&Cropped Images`.
  
## Modules:

- DCGAN Module: `DCGAN.py`   finished  
- Image procseeing module: `image_processing.py`  finished  
- training module: `train.py`  finished
- test module: included in `main.py`      finished  
- coordination module: `main.py`     finished  

## Requirements:  

To run it, you need install `python3` with `Tensorflow`, `numpy` and `scipy`.  
  
## Run：

- Put all CelebA Dataset images in `/dataset`  
- Put your own test images in `/test_img` (number of test images should = batch_size)  
      For example: if you choose batch size = 16 when you train your model, you should put 16 images in `train.py`.  
      Note:
      
      All test images should be 178*218 and face should be in the middle of the image (same as images in the 
      CelebA Dataset). There are some sample test images in /test_img. The test_image will randomly show the
      results of 10(defualt) images in all test images.
      
- Defualt training time is 240 minutes, you can change it in `train.py`  
- To train the model run: `python main.py train`  
- To use your own image to test the model run: `python main.py test`   
- Intermediate results are saved in `/train`   

## Training:  

![training](https://github.com/tangni31/tensorflow/blob/master/project%20code/training.png?raw=true)  
During training, it will show the progress,  remaining training time, batch number, genertor and discriminator's loses.
It will save the intermediate result for every 200(defualt) batches and save the checkpoint for every 10,000(defualt) batches.  
  
## Sample intermediate result:  

This model will randomly choose 1 batch size images(defualt 16 images) in CelebA Dataset as test images to generate intermediate results. Intermediate results only shows 8(defualt) of them.   
left: 16*16 input       
middle: model's output      
right: orignal image  
![sample_intermediate_result](https://github.com/tangni31/tensorflow/blob/master/project%20code/sample_intermediate_result.png?raw=true)


## Test results (test images were downloaded from google.com):

This model was trained on a GTX1080Ti for 4 hours, approximately 70,000 batches.

left: 16*16 input       
middle: model's output      
right: orignal image  

![test result](https://github.com/tangni31/tensorflow/raw/master/project%20code/test_img/test_result.png)
