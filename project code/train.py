import numpy as np
import os.path
import tensorflow as tf
import time
import scipy.misc

train_time = 60 #min
learning_rate_start = 0.0002
learning_rate_half_life = 5000
summary_period = 200
checkpoint_period = 10000


def train_model(train_data):
    td = train_data

    summaries = tf.summary.merge_all()
    td.sess.run(tf.global_variables_initializer())

    lrval = learning_rate_start
    start_time = time.time()
    done = False
    batch = 0

    # Cache test features and labels
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])

    while not done:
        batch += 1
        gene_loss = disc_real_loss = disc_fake_loss = -1.234

        feed_dict = {td.learning_rate: lrval}

        ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]
        _, _, gene_loss, disc_real_loss, disc_fake_loss = td.sess.run(ops, feed_dict=feed_dict)

        if batch % 10 == 0:
            # Show training details
            elapsed = int(time.time() - start_time) / 60
            print('Progress[%3d%%], ETA[%4dm], Batch [%4d], G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f]' %
                  (int(100 * elapsed / train_time), train_time - elapsed,
                   batch, gene_loss, disc_real_loss, disc_fake_loss))

            # Finished or not 
            current_progress = elapsed / train_time
            if current_progress >= 1.0: # Finished
                done = True

            # Update learning rate
            if batch % learning_rate_half_life == 0:
                lrval *= .5

        if batch % summary_period == 0:
            # Show progress with test features
            feed_dict = {td.gene_minput: test_feature}
            gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dict)
            summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')

        if batch % checkpoint_period == 0:
            # Save checkpoint
            save_checkpoint(td, batch)

    save_checkpoint(td, batch)
    print('Finished training!')




def summarize_progress(train_data, feature, label, gene_output, batch, suffix, max_samples=8,test = False):
    td = train_data

    size = [label.shape[1], label.shape[2]]

    low = tf.image.resize_nearest_neighbor(feature, size)
    low = tf.maximum(tf.minimum(low, 1.0), 0.0)

    output = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    image   = tf.concat(axis=2, values=[low, output, label])

    image = image[0:max_samples,:,:,:]
    image = tf.concat(axis=0, values=[image[i,:,:,:] for i in range(max_samples)])
    image = td.sess.run(image)

    filename = 'batch%06d_%s.png' % (batch, suffix)
    if test == False:
        filename = os.path.join('train', filename)
    else:
        filename = os.path.join('test_img', filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))


def save_checkpoint(train_data, batch):
    td = train_data

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join('checkpoint', oldname)
    newname = os.path.join('checkpoint', newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver()
    saver.save(td.sess, newname)

    print(" Checkpoint saved")
