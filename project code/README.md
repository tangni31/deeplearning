Codes for T15's project Image Super-resolution


Dataset: Large-scale CelebFaces Attributes (CelebA) Dataset   http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

DCGAN Module: DCGAN_model.py   finished
Image procseeing module: image_processing.py  finished
training module: train.py   finished
test module: test.py      finished
coordination module: main.py     finished

put CelebA Dataset in /dataset
put your own test images in /test_img (number of test images should = batch_size(default 16))
intermediate results are saved in /train

to train the model run: python main.py train

to use your own image to test the model run: python main.py test


![test result](https://github.com/tangni31/tensorflow/raw/master/project%20code/test_img/test_result.png)