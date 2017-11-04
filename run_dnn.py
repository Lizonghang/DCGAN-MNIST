import tensorflow as tf
import numpy as np
import os
from PIL import Image
from dnn import DNN
from ops import lrelu, linear

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10000, "Epoch to train")
flags.DEFINE_integer("train_size", np.inf, "The allowed max size of train samples")
flags.DEFINE_integer("batch_size", 64, "The size of batch samples")
flags.DEFINE_integer("test_duration", 1, "Duration epoch of testing, 0 for do not test")
flags.DEFINE_float("lr", 0.0001, "Learning rate for optimizer")
flags.DEFINE_string("activation_func", "relu", "Activation function of Deep Neural Network")
flags.DEFINE_string("images_dir", "dcgan_images", "File to predict")
flags.DEFINE_string("checkpoint_dir", "dnn_ckpt", "Directory to save the checkpoints")
flags.DEFINE_string("log_dir", "dnn_logs", "Directory to save te logs")
flags.DEFINE_boolean("train", False, "True for train")
flags.DEFINE_boolean("retrain", False, "True for continue training using images in images_dir")
flags.DEFINE_boolean("training_with_test", True, "Calculate accuracy using testset while training.")
flags.DEFINE_boolean("predict", False, "True for predict")
flags.DEFINE_integer("acc_y", 0, "To calculate accuraccy of generated samples which are type y")
flags.DEFINE_boolean("eval", False, "True for eval")
FLAGS = flags.FLAGS


def main(_):
    assert sum([FLAGS.train, FLAGS.predict, FLAGS.eval, FLAGS.retrain]) == 1

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
        elif FLAGS.predict:
            filename_list = [filename for filename in os.listdir(FLAGS.images_dir) if '.jpg' in filename]

            samples = np.zeros((len(filename_list), 28, 28, 1))

            for i in range(len(filename_list)):
                im = Image.open(os.path.join(FLAGS.images_dir, filename_list[i]))
                im_data = np.asarray(im) / 255.0
                samples[i, :] = np.reshape(im_data, [28, 28, 1])

            predict = dnn.predict(samples)
            print 'Type {}: {}%'.format(FLAGS.acc_y, (predict == FLAGS.acc_y).sum() / float(predict.shape[0]) * 100)

            # err_predict = (predict != FLAGS.acc_y)
            # for i in range(len(filename_list)):
            #     if err_predict[i]:
            #         print 'Error partition:', filename_list[i]
            #     else:
            #         os.remove(os.path.join(FLAGS.images_dir, filename_list[i]))

        elif FLAGS.eval:
            dnn.eval()
        elif FLAGS.retrain:
            dnn.load_network()
            filename_list = [filename for filename in os.listdir(FLAGS.images_dir) if '.jpg' in filename]
            max_batch = len(filename_list) / FLAGS.batch_size
            assert max_batch != 0
            for i in range(max_batch):
                samples = np.zeros((FLAGS.batch_size, 28, 28, 1))
                labels = np.zeros((FLAGS.batch_size, 10))

                files = filename_list[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                for j in range(FLAGS.batch_size):
                    im = Image.open(os.path.join(FLAGS.images_dir, files[j]))
                    im_data = np.asarray(im) / 255.0
                    samples[j, :] = np.reshape(im_data, [28, 28, 1])
                    labels[j, int(files[j].split('_')[-2])] = 1
                    os.remove(os.path.join(FLAGS.images_dir, files[j]))

                dnn.refit(samples, labels)


if __name__ == '__main__':
    """
    Fit:        python run_dnn.py --train
    Predict:    python run_dnn.py --predict --acc_y 0
    Eval:       python run_dnn.py --eval
    Retrain:    python run_dnn.py --retrain
    """
    tf.app.run()
