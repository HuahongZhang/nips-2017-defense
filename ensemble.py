"""Implementation of sample defense.

This defense loads inception resnet v2 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread

import tensorflow as tf
import inception_resnet_v2
from tensorflow.contrib.slim.nets import inception
import csv

import cv2

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'ckpt1', '', 'Path to checkpoint 1 for inception network.')

tf.flags.DEFINE_string(
    'ckpt2', '', 'Path to checkpoint 2 for inception network.')

tf.flags.DEFINE_string(
    'ckpt3', '', 'Path to checkpoint 3 for inception network.')

tf.flags.DEFINE_string(
    'ckpt4', '', 'Path to checkpoint 4 for inception network.')

tf.flags.DEFINE_string(
    'ckpt5', '', 'Path to checkpoint 5 for inception network.')

tf.flags.DEFINE_float(
    'weight1', 0.23, 'Weight of network 1')

tf.flags.DEFINE_float(
    'weight2', 0.21, 'Weight of network 2')

tf.flags.DEFINE_float(
    'weight3', 0.20, 'Weight of network 3')

tf.flags.DEFINE_float(
    'weight4', 0.18, 'Weight of network 4')

tf.flags.DEFINE_float(
    'weight5', 0.18, 'Weight of network 5')

tf.flags.DEFINE_string(
    'network', '', 'Which kind of network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'dataset', 'dev_dataset.csv', 'The dataset file.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


class DatasetMetadata(object):
    def __init__(self, filename):
        """Initializes instance of DatasetMetadata."""
        self._true_labels = {}
        self._target_classes = {}
        with open(filename) as f:
            reader = csv.reader(f)
            header_row = next(reader)
            try:
                row_idx_image_id = header_row.index('ImageId')
                row_idx_true_label = header_row.index('TrueLabel')
                row_idx_target_class = header_row.index('TargetClass')
            except ValueError:
                raise IOError('Invalid format of dataset metadata.')
            for row in reader:
                if len(row) < len(header_row):
                    # skip partial or empty lines
                    continue
                try:
                    image_id = row[row_idx_image_id]
                    self._true_labels[image_id] = int(row[row_idx_true_label])
                    self._target_classes[image_id] = int(row[row_idx_target_class])
                except (IndexError, ValueError):
                    raise IOError('Invalid format of dataset metadata')

    def get_true_label(self, image_id):
        """Returns true label for image with given ID."""
        return self._true_labels[image_id]

    def get_target_class(self, image_id):
        """Returns target class for image with given ID."""
        return self._target_classes[image_id]

    def save_target_classes(self, filename):
        """Saves target classed for all dataset images into given file."""
        with open(filename, 'w') as f:
            for k, v in self._target_classes.items():
                f.write('{0}.png,{1}\n'.format(k, v))


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    images_denoised = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image_org = imread(f, mode='RGB').astype(np.uint8)
            image_denoised = cv2.bilateralFilter(image_org, 9, 35, 35).astype(np.float) / 255.0
            # image_denoised = image_org.astype(np.float) / 255.0
            image = image_org.astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        images_denoised[idx, :, :, :] = image_denoised* 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images, images_denoised
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images, images_denoised

# dataset_meta = DatasetMetadata(FLAGS.dataset)


def get_res(network_type, checkpoint_path):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        if network_type == 'v2':
            with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
                _, end_points = inception_resnet_v2.inception_resnet_v2(
                    x_input, num_classes=num_classes, is_training=False)
        else:
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                _, end_points = inception.inception_v3(
                    x_input, num_classes=num_classes, is_training=False)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())

        res_dict = {}
        res_dict_denoised = {}
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            for filenames, images, images_denoised in load_images(FLAGS.input_dir, batch_shape):
                labels = sess.run(predicted_labels, feed_dict={x_input: images})
                labels_denoised = sess.run(predicted_labels, feed_dict={x_input: images_denoised})
                for filename, label, label_denoised in zip(filenames, labels, labels_denoised):
                    res_dict[filename] = label
                    res_dict_denoised[filename] = label_denoised

        return res_dict, res_dict_denoised


def main(_):
    checkpoint_path1 = FLAGS.ckpt1
    checkpoint_path2 = FLAGS.ckpt2
    checkpoint_path3 = FLAGS.ckpt3
    checkpoint_path4 = FLAGS.ckpt4
    checkpoint_path5 = FLAGS.ckpt5
    res_dict1, res_dict_denoised1 = get_res('v2', checkpoint_path1)
    res_dict2, res_dict_denoised2 = get_res('v3', checkpoint_path2)
    res_dict3, res_dict_denoised3 = get_res('v3', checkpoint_path3)
    res_dict4, res_dict_denoised4 = get_res('v3', checkpoint_path4)
    res_dict5, res_dict_denoised5 = get_res('v3', checkpoint_path5)
    with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
        for i in res_dict1.keys():
            final_res = np.zeros(1001)
            final_res[res_dict1[i]] += FLAGS.weight1
            final_res[res_dict2[i]] += FLAGS.weight2
            final_res[res_dict3[i]] += FLAGS.weight3
            final_res[res_dict4[i]] += FLAGS.weight4
            final_res[res_dict5[i]] += FLAGS.weight5
            final_res[res_dict_denoised1[i]] += FLAGS.weight1 / 1.5
            final_res[res_dict_denoised2[i]] += FLAGS.weight2 / 1.5
            final_res[res_dict_denoised3[i]] += FLAGS.weight3 / 1.5
            final_res[res_dict_denoised4[i]] += FLAGS.weight4 / 1.5
            final_res[res_dict_denoised5[i]] += FLAGS.weight5 / 1.5
            res = np.argmax(final_res)
            out_file.write('{0},{1}\n'.format(i, res))
            # print(res, res_dict1[i], res_dict2[i], res_dict3[i], res_dict4[i], res_dict5[i], res_dict_denoised1[i])


if __name__ == '__main__':
    tf.app.run()
