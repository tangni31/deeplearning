import numpy as np
import tensorflow as tf


batch_size = 16
gen_l1_factor = 0.9
beta1 = 0.5 #arXiv:1511.06434v2 β1 for adam optimizer
learning_rate = 0.0002 #arXiv:1511.06434v2

class Model:
    
    def __init__(self, name, features):
        self.name = name
        self.outputs = [features]

    def get_layer_name(self, layer=None):
        if layer is None:
            layer = self.get_num_layers()
        return '%s_L%03d' % (self.name, layer+1)

    def get_num_inputs(self):
        return int(self.get_output().get_shape()[-1])

    def get_num_layers(self):
        return len(self.outputs)

    def add_batch_norm(self, scale=False):
        with tf.variable_scope(self.get_layer_name()):
            out = tf.contrib.layers.batch_norm(self.get_output(), scale=scale)
        self.outputs.append(out)
        return self

    def add_tanh(self):
        with tf.variable_scope(self.get_layer_name()):
            out = tf.nn.tanh(self.get_output())
        self.outputs.append(out)
        return self

    def add_relu(self):
        with tf.variable_scope(self.get_layer_name()):
            out = tf.nn.relu(self.get_output())
        self.outputs.append(out)
        return self

    def add_leakrelu(self,leak=0.2):
        with tf.variable_scope(self.get_layer_name()):
            out = tf.nn.leaky_relu(self.get_output(),leak)
        self.outputs.append(out)
        return self

    def glorot_initializer(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.
        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],mean=0.0, stddev=stddev)

    def add_conv2d(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        with tf.variable_scope(self.get_layer_name()):
            prev_units = self.get_num_inputs()
            # Weight term and convolution
            initw = self.glorot_initializer(prev_units, num_units, mapsize, stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            out = tf.nn.conv2d(self.get_output(), weight, strides=[1, stride, stride, 1], padding='SAME')
            # Bias term
            initb = tf.constant(0.0, shape=[num_units])
            bias  = tf.get_variable('bias', initializer=initb)
            out = tf.nn.bias_add(out, bias)
        self.outputs.append(out)
        return self

    def add_conv2d_transpose(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        with tf.variable_scope(self.get_layer_name()):
            prev_units = self.get_num_inputs()
            # Weight term and convolution
            initw = self.glorot_initializer(prev_units, num_units, mapsize, stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            prev_output = self.get_output()
            output_shape = [batch_size,
                            int(prev_output.get_shape()[1]) * stride,
                            int(prev_output.get_shape()[2]) * stride,
                            num_units]
            out = tf.nn.conv2d_transpose(self.get_output(), weight, output_shape=output_shape,
                                            strides=[1, stride, stride, 1], padding='SAME')
            # Bias term
            initb = tf.constant(0.0, shape=[num_units])
            bias = tf.get_variable('bias', initializer=initb)
            out = tf.nn.bias_add(out, bias)
        self.outputs.append(out)
        return self

    def add_residual_block(self, num_units, mapsize=3, num_layers=2, stddev_factor=1e-3):
        """Adds a residual block as per Arxiv 1512.03385, Figure 3"""
        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]):
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=1.)
        bypass = self.get_output()
        # Residual block
        for _ in range(num_layers):
            self.add_batch_norm()
            self.add_relu()
            self.add_conv2d(num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
        self.add_sum(bypass)
        return self

    def add_sum(self, term):
        """Adds a layer that sums the top layer with the given term"""
        with tf.variable_scope(self.get_layer_name()):
            prev_shape = self.get_output().get_shape()
            term_shape = term.get_shape()
            assert prev_shape == term_shape and "Can't sum terms with a different size"
            out = tf.add(self.get_output(), term)
        self.outputs.append(out)
        return self

    def add_mean(self):
        """Adds a layer that averages the inputs from the previous layer"""
        with tf.variable_scope(self.get_layer_name()):
            prev_shape = self.get_output().get_shape()
            reduction_indices = list(range(len(prev_shape)))
            assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
            reduction_indices = reduction_indices[1:-1]
            out = tf.reduce_mean(self.get_output(), reduction_indices=reduction_indices)
        self.outputs.append(out)
        return self

    def add_upscale(self):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""
        prev_shape = self.get_output().get_shape()
        size = [2 * int(s) for s in prev_shape[1:3]]
        out  = tf.image.resize_nearest_neighbor(self.get_output(), size)
        self.outputs.append(out)
        return self        

    def get_output(self):
        """Returns the output from the topmost layer of the network"""
        return self.outputs[-1]


def _discriminator_model(sess, features, disc_input):
    # using convolution layers to replace pooling and fully connected layer arXiv:1511.06434v2
    mapsize = 3
    layers = [64, 128, 256, 512]

    old_vars = tf.global_variables()

    model = Model('DIS', 2*disc_input - 1)
    stddev_factor = 2.0
    for layer in range(len(layers)):
        nunits = layers[layer]


        model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        model.add_batch_norm()# Use batchnorm in both the generator and the discriminator arXiv:1511.06434v2
        model.add_leakrelu()#Use LeakyReLU activation in the discriminator for all layers: arXiv:1511.06434v2

    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_leakrelu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_leakrelu()
    # Linearly map to real/fake and return average score
    model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_mean()
    out = model.get_output()

    new_vars = tf.global_variables()
    disc_vars = list(set(new_vars) - set(old_vars))

    return out, disc_vars

def _generator_model(sess, features, labels, channels):
    # Upside-down all-convolutional resnet
    mapsize = 3
    res_units  = [256, 128, 96]
    old_vars = tf.global_variables()
    #Arxiv 1603.05027
    model = Model('GEN', features)

    for ru in range(len(res_units)-1):
        nunits  = res_units[ru]
        for j in range(2):
            model.add_residual_block(nunits, mapsize=mapsize)
        model.add_upscale()
        model.add_batch_norm()
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)

    #using convolution layers to replace pooling and fully connected layer arXiv:1511.06434v2
    nunits = res_units[-1]

    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
    model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.) #fully connected layers is replaced by simple 1-by-1 convolutions.
    model.add_relu()

    model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
    model.add_tanh()#tanh for output layer
    
    new_vars  = tf.global_variables()
    gene_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), gene_vars

def create_model(sess, features, labels):
    # Generator
    rows = int(features.get_shape()[1])
    cols = int(features.get_shape()[2])
    channels = int(features.get_shape()[3])

    gene_minput = tf.placeholder(tf.float32, shape=[batch_size, rows, cols, channels])

    with tf.variable_scope('gene') as scope:
        gene_output, gene_var_list = \
                    _generator_model(sess, features, labels, channels)

        scope.reuse_variables()

        gene_moutput, _ = _generator_model(sess, gene_minput, labels, channels)
    
    # Discriminator with real data
    disc_real_input = tf.identity(labels, name='disc_real_input')

    with tf.variable_scope('disc') as scope:
        disc_real_output, disc_var_list = \
                _discriminator_model(sess, features, disc_real_input)

        scope.reuse_variables()
            
        disc_fake_output, _ = _discriminator_model(sess, features, gene_output)

    return [gene_minput,      gene_moutput,
            gene_output,      gene_var_list,
            disc_real_output, disc_fake_output, disc_var_list]

def _downscale(images, K):
    '''
    downscale the generator's 64*64 output to 16*16 to calculate generator lost 1
    downscale using a convolution layer
    '''
    '''
        filter parameters: [   1/16, 1/16, 1/16, 1/16
                                1/16, 1/16, 1/16, 1/16
                                1/16, 1/16, 1/16, 1/16
                                1/16, 1/16, 1/16, 1/16  ]  
        '''
    arr = np.zeros([K, K, 3, 3])
    arr[:,:,0,0] = 1.0/(K*K)
    arr[:,:,1,1] = 1.0/(K*K)
    arr[:,:,2,2] = 1.0/(K*K)
    dowscale_weight = tf.constant(arr, dtype=tf.float32)
    
    downscaled = tf.nn.conv2d(images, dowscale_weight,
                              strides=[1, K, K, 1],
                              padding='SAME')
    return downscaled

def generator_loss(disc_output, gene_output, features):
    # lost 2, generator output fool discriminator or not, generator wants discriminator's output close to 1
    # use cross entropy, feature: discriminator's output, labels = 1
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_output, labels=tf.ones_like(disc_output))
    gene_loss2 = tf.reduce_mean(cross_entropy, name='gene_loss2')


    K = int(gene_output.get_shape()[1])//int(features.get_shape()[1])
    assert K == 2 or K == 4 or K == 8    
    downscaled = _downscale(gene_output, K)
    # lost 1, difference between generator output and input image
    gene_loss1 = tf.reduce_mean(tf.abs(downscaled - features), name='gene_loss1')

    gene_loss = tf.add((1.0 - gen_l1_factor) * gene_loss2,
                           gen_l1_factor * gene_loss1, name='gene_loss')
    
    return gene_loss

def discriminator_loss(disc_real_output, disc_fake_output):
    '''
            :param dis_real_output: discriminator's output when input is a real image from dataset
            :param dis_fake_output: discriminator's output when input is a fake image generated by generator
            '''
    # discriminator want its output close to 1 if input is a real image
    cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_output, labels=tf.ones_like(disc_real_output))
    disc_real_loss = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')
    
    cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_output, labels=tf.zeros_like(disc_fake_output))
    disc_fake_loss = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')

    return disc_real_loss, disc_fake_loss

def optimizers(gene_loss, gene_var_list,
                      disc_loss, disc_var_list):
    '''
        arXiv:1511.06434v2:"we used the Adam optimizer with tuned hyperparameters,
                            We found the suggested learning rate of 0.001, to be too high, using 0.0002 instead.
                            momentum term β1 at 0.5 helped stabilize training"
        '''
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    
    gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, name='gene_optimizer')
    disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, name='disc_optimizer')

    gene_minimize = gene_opti.minimize(gene_loss, var_list=gene_var_list, name='gene_loss_minimize', global_step=global_step)
    disc_minimize = disc_opti.minimize(disc_loss, var_list=disc_var_list, name='disc_loss_minimize', global_step=global_step)
    
    return (global_step, learning_rate, gene_minimize, disc_minimize)
