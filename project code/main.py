import numpy as np
import os.path
import tensorflow as tf
import DCGAN
import image_processing
import train
import random
from sys import argv

random_seed = 0#Seed used to initialize rng
batch_size = 16

def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists('checkpoint'):
        tf.gfile.MakeDirs('checkpoint')

    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists('train'):
            tf.gfile.DeleteRecursively('train_dir')
        tf.gfile.MakeDirs('train_dir')

    # Return names of training files
    if not tf.gfile.Exists('dataset') or \
            not tf.gfile.IsDirectory('dataset'):
        raise FileNotFoundError("Could not find folder `%s'" % ('dataset',))

    filenames = tf.gfile.ListDirectory('dataset')
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join('dataset', f) for f in filenames]
    return filenames


def setup_tensorflow():
    # Create session
    config = tf.ConfigProto(log_device_placement=False)
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    summary_writer = tf.summary.FileWriter('train', sess.graph)
    return sess, summary_writer


class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def training():
    # Prepare directories
    all_filenames = prepare_dirs(delete_train_dir=True)
    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()
    # Separate training and test sets
    train_filenames = all_filenames[:-batch_size]
    test_filenames = all_filenames[-batch_size:]
    # Setup async input queues
    train_features, train_labels = image_processing.get_inputs(sess, train_filenames)
    test_features, test_labels = image_processing.get_inputs(sess, test_filenames)
    # Add some noise during training
    noise_level = .03
    noisy_train_features = train_features + \
                           tf.random_normal(train_features.get_shape(), stddev=noise_level)
    # Create and initialize model
    #gene_minput and gene_moutput is used for testing
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
        DCGAN.create_model(sess, noisy_train_features, train_labels)
    gene_loss = DCGAN.generator_loss(disc_fake_output, gene_output, train_features)
    disc_real_loss, disc_fake_loss = \
        DCGAN.discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')

    (global_step, learning_rate, gene_minimize, disc_minimize) = \
        DCGAN.optimizers(gene_loss, gene_var_list,
                                     disc_loss, disc_var_list)
    # Train model
    train_data = TrainData(locals())
    train.train_model(train_data)


def testing():#testing allowed user use his own image to test the trained model
    sess, summary_writer = setup_tensorflow()
    # Prepare directories
    Tfilenames = prepare_dirs(delete_train_dir=False)
    features, labels = image_processing.get_inputs(sess, Tfilenames)
    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
        DCGAN.create_model(sess, features, labels)
    # Restore variables from checkpoint_dir
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join('checkpoint', filename)
    saver.restore(sess, filename)
    print("Restore model")
    # region prediction test
    predict_restore = gene_moutput
    # Prepare directories
    test_filenames = tf.gfile.ListDirectory('test_img') #number of images in 'test_img' should = batch size
    test_filenames = sorted(test_filenames)
    random.shuffle(test_filenames)
    test_filenames = [os.path.join('test_img', f) for f in filenames]
    test_features, test_labels = image_processing.get_inputs(sess, test_filenames)
    test_img4_input, test_img4_original = sess.run([test_features, test_labels])
    feed_dict = {gene_minput: test_img4_input}
    prob = sess.run(predict_restore, feed_dict)
    td = TrainData(locals())
    # max_samples=10, test image will output 10 results
    train.summarize_progress(td, test_img4_input, test_img4_original, prob, 3, 'out', max_samples=10, test=True)
    print("Finish testing")


if __name__ == '__main__':

    if argv[1] == 'train':
        training()
    elif argv[1] == 'test':
        testing()
    else:
         print("Invalid input! You must run 'python main.py test' to test your model or 'python main.py train' to train your model")
