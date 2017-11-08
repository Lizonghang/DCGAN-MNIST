from dcgan import DCGAN
from ops import *
import tensorflow as tf
import numpy as np
import random
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train")
flags.DEFINE_float("lr", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam")
flags.DEFINE_integer("z_dim", 100, "Number of noise")
flags.DEFINE_integer("train_size", np.inf, "The size of train images")
flags.DEFINE_integer("batch_size", 64, "The size of batch images")
flags.DEFINE_string("dataset", "dataset", "The path of dataset")
flags.DEFINE_string("checkpoint_dir", "dcgan_ckpt", "Directory name to save the checkpoints")
flags.DEFINE_string("images_dir", "dcgan_images", "Directory name to save the images generated")
flags.DEFINE_string("logs_dir", "dcgan_logs", "Directory name to save the logs")
flags.DEFINE_boolean("train", True, "True for training, False for generating [False]")
flags.DEFINE_integer("gen_size", 64, "Num to generate")
flags.DEFINE_integer("gen_y", None, "Type for generating")
FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.images_dir):
        os.makedirs(FLAGS.images_dir)
    if not os.path.exists(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)

    with tf.Session() as sess:
        dcgan = DCGAN(
            sess,
            lr=FLAGS.lr,
            beta1=FLAGS.beta1,
            epoch=FLAGS.epoch,
            batch_size=FLAGS.batch_size,
            train_size=FLAGS.train_size,
            z_dim=FLAGS.z_dim,
            dataset=FLAGS.dataset,
            checkpoint_dir=FLAGS.checkpoint_dir,
            logs_dir=FLAGS.logs_dir,
            images_dir=FLAGS.images_dir
        )

        if FLAGS.train:
            dcgan.train()
        else:
            counter = 1
            for j in range(FLAGS.gen_size / FLAGS.batch_size):
                gen_y = FLAGS.gen_y or [random.randint(0, 9) for _ in range(FLAGS.batch_size)]
                samples = dcgan.generate(gen_y)
                for i in range(samples.shape[0]):
                    filepath = './{}/gen_samples_{}_{}.jpg'.format(FLAGS.images_dir, FLAGS.gen_y or gen_y[i], counter)
                    save_images(samples[i, :].reshape(1, 28, 28, 1), (1, 1), filepath)
                    print 'Image gen_samples_{}_{}.jpg saved to {}'.format(FLAGS.gen_y or gen_y[i], counter, FLAGS.images_dir)
                    counter += 1


if __name__ == '__main__':
    """
    Fit: python run_dcgan.py
    Generate: python run_dcgan.py --train False --gen_size 64
    """
    tf.app.run()
