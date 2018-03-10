import tensorflow as tf

batch_size = 16

class Model:
    #layer has 4 dimensions: (batch, width, height, channels)
    def __init__(self,name,image_input): #name is the model's name eg. generator, discriminator
        self.name = name
        self.output = [image_input] #store input in a list and calculate output

    def get_output(self):
        return self.output[-1]

    def get_layer_name(self, layer=None):
        #generate a name for the layer
        if layer is None:
            layer = self.get_layer_num()
        return '%s_L%03d' % (self.name, layer + 1)

    def get_layer_num(self):
        return len(self.output)

    def get_input_num(self):#return number of units in last layer
        return int(self.get_output().get_shape()[-1])

    def tanh(self):
        #sigmoid function
        with tf.variable_scope(self.get_layer_name()):
            output = tf.nn.tanh(self.get_output())
        self.output.append(output)

    def relu(self):
        #relu function
        with tf.variable_scope(self.get_layer_name()):
            out = tf.nn.relu(self.get_output())
        self.output.append(out)


    def upscale(self):
        #generator need to upscale input images from 16*16 to 64*64
        #this upscale layer can upscale output by 2x
        input_shape = self.get_output().get_shape() #shape of previous layer
        new_shape = [input_shape[0], 2*int(input_shape[1]), 2*int(input_shape[2]), input_shape[3]]#(batch, width*2, height*2, channels)
        output = tf.image.resize_nearest_neighbor(self.get_output(), new_shape)
        self.output.append(output)
        return self

    def leaky_relu(self,leak=0.2): #Use LeakyReLU activation in the discriminator for all layers: arXiv:1511.06434v2
        with tf.variable_scope(self.get_layer_name()):
            t1 = 0.5 * (1 + leak)
            t2 = 0.5 * (1 - leak)
            out = t1 * self.get_output() + \
                  t2 * tf.abs(self.get_output())
        self.output.append(out)
        return self

    def batchnorm(self,scale=False):
        # Use batchnorm in both the generator and the discriminator arXiv:1511.06434v2
        with tf.variable_scope(self.get_layer_name()):
            out = tf.contrib.layers.batch_norm(self.get_output(), scale=scale)
            '''scale: If True, multiply by gamma. If False, gamma is not used. When the next layer is linear 
                (also e.g. nn.relu), this can be disabled since the scaling can be done by the next layer.'''
        self.output.append(out)
        return self


    def mean(self):
        """averages the inputs from the previous layer"""

        with tf.variable_scope(self.get_layer_name()):
            prev_shape = self.get_output().get_shape()
            reduction_indices = list(range(len(prev_shape)))
            reduction_indices = reduction_indices[1:-1]
            out = tf.reduce_mean(self.get_output(), reduction_indices=reduction_indices)

        self.output.append(out)
        return self

    def conv2d(self, num_units, mapsize=1,stride=1):
        #a convolution layer
        with tf.variable_scope(self.get_layer_name()):
            prev_units = self.get_input_num()#number of units in previous layer
            weight_initial = tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                    mean=0.0, stddev=0.02)
            weight = tf.get_variable('weight', initializer=weight_initial)
            out = tf.nn.conv2d(self.get_output(), weight,        #tf.nn.conv2d(input, filter, strides, padding)
                           strides=[1, stride, stride, 1],         # filter: [filter_height, filter_width, in_channels, out_channels]
                           padding='SAME')

        self.output.append(out)
        return self

    def transpose_conv2d(self,num_units, mapsize=1,stride=1):
        with tf.variable_scope(self.get_layer_name()):
            prev_units = self.get_input_num()#number of units in previous layer

            weight_initial = tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                                 mean=0.0, stddev=0.02)
            weight = tf.get_variable('weight', initializer=weight_initial)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2]) #[mapsize, mapsize, num_units, prev_units]
            input = self.get_output()#output from previous layer
            output_shape = [batch_size, int(input.get_shape()[1]) * stride,
                            int(input.get_shape()[2]) * stride, num_units]
            out = tf.nn.conv2d_transpose(self.get_output(), weight, output_shape=output_shape, #conv2d_transpose(value, filter, output_shape, strides, padding)
                                         strides=[1, stride, stride, 1], padding='SAME') #filter:[filter_height, filter_width, out_channels, in_channels]



def generator(sess, image_input, channels=3):
    mapsize = 3
    layers = [96,128,256]
    old_vars = tf.global_variables()

    model = Model('generator',image_input)

    for l in range(len(layers)-1):
        units = layers[l]


        model.upscale()
        model.batchnorm()
        model.relu()
        model.transpose_conv2d(units, mapsize=mapsize, stride=1)

    #using convolution layers to replace pooling and fully connected layer arXiv:1511.06434v2
    units = layers[-1]

    model.conv2d(units, mapsize=mapsize, stride=1)
    model.relu()

    model.conv2d(units, mapsize=1, stride=1)
    model.relu()

    model.conv2d(channels, mapsize=1, stride=1)
    model.tanh()#tanh for output layer

    new_vars = tf.global_variables()
    gen_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), gen_vars

def discriminator(sess, disc_input):
    mapsize = 3
    layers = [64, 128, 256, 512]
    old_vars = tf.global_variables()
    
    model = Model('discriminator', 2 * disc_input - 1)

    for l in range(len(layers)):
        units = layers[l]

        model.conv2d(num_units=units, mapsize=mapsize,stride=2)
        model.batchnorm()
        model.leaky_relu()

    model.conv2d(num_units=units, mapsize=mapsize, stride=1)
    model.batchnorm()
    model.leaky_relu()

    model.conv2d(num_units=units, mapsize=1, stride=1)
    model.batchnorm()
    model.leaky_relu()

    model.conv2d(1, mapsize=1, stride=1)

    new_vars = tf.global_variables()
    dis_vars = list(set(new_vars) - set(old_vars))
    return model.get_output(), dis_vars

def create_model(sess, image_input, image_output):
    #create generator model
    with tf.variable_scope('generator') as scope:
        gen_output, gen_vars = generator(sess, image_input)
        scope.reuse_variables()

    # Discriminator with real input
    dis_real_input = tf.identity(image_output, name="dis_real_input")
    with tf.variable_scope('discriminator') as scope:
        dis_real_output, dis_vars = discriminator(sess, dis_real_input)#real 64*64 image as discriminator's input
        scope.reuse_variables()
        dis_fake_output, _ = discriminator(sess,gen_output)#generator's output as discriminator's input        
    return [gen_output, gen_vars, dis_real_output, dis_fake_output, dis_vars]

def downscale(gen_output):
    '''
    downscale the generator's 64*64 output to 16*16 to calculate generator lost 1
    downscale using a convolution layer
    '''
    weight = np.zeros([4,4,3,3])#filter: [filter_height=4, filter_width=4, in_channels=3, out_channels=4]
    '''
    filter parameters: [   1/16, 1/16, 1/16, 1/16
                            1/16, 1/16, 1/16, 1/16
                            1/16, 1/16, 1/16, 1/16
                            1/16, 1/16, 1/16, 1/16  ]  
    '''
    weight[:, :, 0, 0] = 1.0 / 16
    weight[:, :, 1, 1] = 1.0 / 16
    weight[:, :, 2, 2] = 1.0 / 16
    weight = tf.constant(weight, dtype=tf.float32)
    downscale_image = tf.nn.conv2d(gen_output, weight, strides=[1,4,4,1], padding='SAME')
    return downscale_image

def generator_loss(dis_output, gen_output, image_input):
    #lost 1, difference between generator output and input image
    downscale_gen_output = downscale(gen_output)
    gen_loss1 = tf.reduce_mean(tf.abs(downscale_gen_output - image_input), name='gen_loss1')

    #lost 2, generator output fool discriminator or not, generator wants discriminator's output close to 1
    #use cross entropy, feature: discriminator's output, labels = 1
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_output, labels=tf.ones_like(dis_output))
    gen_loss2 = tf.reduce_mean(cross_entropy, name='gen_loss2')


    gen_loss = tf.add(gen_loss1*gen_loss1_factor, gen_loss2*(1-gen_loss1_factor))

    return gen_loss

def discriminator_loss(dis_real_output, dis_fake_output):
    '''
    :param dis_real_output: discriminator's output when input is a real image from dataset
    :param dis_fake_output: discriminator's output when input is a fake image generated by generator
    '''
    #discriminator want its output close to 1 if input is a real image
    cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_output,
                                                                 labels=tf.ones_like(dis_real_output))
    dis_real_loss = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')

    # discriminator want its output close to 0 if input is a fake image
    cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_output,
                                                                 labels=tf.zeros_like(dis_fake_output))
    dis_fake_loss = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')
    return  dis_real_loss, dis_fake_loss

def optimizers(gen_loss, gen_vars, dis_loss, dis_vars):
    '''
    arXiv:1511.06434v2:"we used the Adam optimizer with tuned hyperparameters,
                        We found the suggested learning rate of 0.001, to be too high, using 0.0002 instead.
                        momentum term β1 at 0.5 helped stabilize training"
    '''
    
        
