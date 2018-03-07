import tensorflow as tf

def get_inputs(sess, filenames): #filenames is a list contain all images' name that need to be processed
    #read images
    reader = tf.WholeFileReader()
    try:
        filename_list = tf.train.string_input_producer(filenames)
        key, value = reader.read(filename_list)
        image = tf.image.decode_jpeg(value, channels=3, name="dataset_image")
    except:
        file = open(filenames. 'rb')
        data = file.read()
        file.close()
        channels = 3
        image = tf.image.decode_jpeg(data, channels=3, name="dataset_image")
    image.set_shape([None, None, 3])

    # image processing
    #random processing
    image = tf.image.random_flip_left_right(image) #random filp image
    image = tf.image.random_saturation(image, .8, 1.2)#random adjust saturation
    image = tf.image.random_brightness(image, .1) #random adjust brightness
    image = tf.image.random_contrast(image, .8, 1.2)#random adjust contrast

    #random crop
    crop_size_1 = 144
    crop_size_2 = 128
    off_x, off_y =25-8, 60-8 #make sure not cut the face
    image =  tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_1, crop_size_1) #144*144
    image = tf.random_crop(image, [crop_size_2, crop_size_2, 3]) #128*128
    image = tf.reshape(image, [1, crop_size_2, crop_size_2, 3])
    image = tf.cast(image, tf.float32)/255.0

    #downsample
    image = tf.image.resize_area(image, [64, 64])
    downsample = tf.image.resize_area(image, [16, 16])
    image_output = tf.reshape(image, [64, 64,3]) #64*64 desire output
    image_input = tf.reshape(downsample, [16, 16,3])#16*16 input

    # batch
    batch_size = 16
    image_inputs, image_outputs = tf.train.batch([image_input, image_output],
                                      batch_size=batch_size, num_threads=4,capacity = batch_size*3,
                                      name='inputs_and_outputs')
    tf.train.start_queue_runners(sess=sess)
    return image_inputs, image_outputs

'''
# show results in tensorboard

output_image_summary = tf.summary.image('output image', tf.expand_dims(image_output, 0))
input_image_summary = tf.summary.image('input image', tf.expand_dims(image_input, 0))
merged = tf.summary.merge_all()

sess = tf.Session()
writer = tf.summary.FileWriter('temp')

summary = sess.run(merged)
writer.add_summary(summary)

writer.close()
sess.close()
'''
