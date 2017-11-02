import tensorflow as tf
import numpy as np
import os

from dnn import DNN
from ops import lrelu, linear

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Epoch to train")
flags.DEFINE_integer("train_size", np.inf, "The allowed max size of train samples")
flags.DEFINE_integer("batch_size", 64, "The size of batch samples")
flags.DEFINE_integer("test_duration", 1, "Duration epoch of testing, 0 for do not test")
flags.DEFINE_integer("lr_decay_step", 1, "Step of learning rate decay, in epoch")
flags.DEFINE_float("lr", 0.1, "Learning rate for optimizer")
flags.DEFINE_float("lr_decay_rate", 0.1, "Exponential decay rate of learning rate")
flags.DEFINE_string("activation_func", "relu", "Activation function of Deep Neural Network")
flags.DEFINE_string("predictset", "dataset/gen_samples.csv", "File to predict")
flags.DEFINE_string("checkpoint_dir", "dnn_ckpt", "Directory to save the checkpoints")
flags.DEFINE_string("log_dir", "dnn_logs", "Directory to save te logs")
flags.DEFINE_boolean("train", True, "True for training, False for predicting")
flags.DEFINE_boolean("training_with_test", True, "Calculate accuracy using testset while training.")
flags.DEFINE_boolean("predict_with_label", False, "Whether the data to predict has label")
flags.DEFINE_integer("acc_y", 0, "To calculate accuraccy of generated samples which are type y")
FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    if FLAGS.activation_func == 'sigmoid':
        activation_func = tf.nn.sigmoid
    elif FLAGS.activation_func == 'tanh':
        activation_func = tf.nn.tanh
    elif FLAGS.activation_func == 'relu':
        activation_func = tf.nn.relu
    elif FLAGS.activation_func == 'leakrelu':
        activation_func = lrelu
    else:
        activation_func = linear

    with tf.Session() as sess:
        dnn = DNN(sess,
                  FLAGS.lr,
                  FLAGS.lr_decay_rate,
                  FLAGS.lr_decay_step,
                  FLAGS.batch_size,
                  FLAGS.train_size,
                  FLAGS.epoch,
                  FLAGS.test_duration,
                  activation_func,
                  FLAGS.checkpoint_dir,
                  FLAGS.log_dir
                  )
        if FLAGS.train:
            dnn.fit(training_with_test=FLAGS.training_with_test)
        else:
            predict = dnn.predict(FLAGS.predictset, FLAGS.predict_with_label)
            print 'Type {}: {}%'.format(FLAGS.acc_y, (predict == FLAGS.acc_y).sum() / float(predict.shape[0]) * 100)


if __name__ == '__main__':
    """
    Fit:python run_dnn.py
    Predict:python run_dnn.py --train False
    """
    tf.app.run()
