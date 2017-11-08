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
flags.DEFINE_boolean("filter", False, "Filter generated images")
flags.DEFINE_boolean("eval", False, "True for eval")
flags.DEFINE_boolean("eval_G", False, "True for eval the images generated by G network")
flags.DEFINE_string("eval_G_dir", "dcgan_eval", "Directory to save the images generated by G network")
FLAGS = flags.FLAGS


def main(_):
    assert sum([FLAGS.train, FLAGS.predict, FLAGS.eval, FLAGS.retrain, FLAGS.eval_G, FLAGS.filter]) == 1

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.eval_G_dir):
        os.makedirs(FLAGS.eval_G_dir)

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
        elif FLAGS.predict or FLAGS.filter:
            filename_list = [filename for filename in os.listdir(FLAGS.images_dir) if '.jpg' in filename]

            samples = np.zeros((len(filename_list), 28, 28, 1))
            gen_y = []

            for i in range(len(filename_list)):
                im = Image.open(os.path.join(FLAGS.images_dir, filename_list[i]))
                im_data = np.asarray(im) / 255.0
                samples[i, :] = np.reshape(im_data, [28, 28, 1])
                gen_y.append(int(filename_list[i].split('_')[-2]))

            dnn.load_network()
            predict = dnn.predict(samples)
            if FLAGS.predict:
                print 'Accuracy: {}%'.format((predict == gen_y).sum() / float(predict.shape[0]) * 100)
            else:
                err_predict = (predict != gen_y)
                counter = 0
                for i in range(len(filename_list)):
                    if err_predict[i]:
                        # if gen_y[i] in [0, 1, 2, 3, 4] and predict[i] in [5, 6, 7, 8, 9]:
                        #     print 'Error type 0->1:', filename_list[i]
                        #     counter += 1
                        # elif gen_y[i] in [5, 6, 7, 8, 9] and predict[i] in [0, 1, 2, 3, 4]:
                        #     print 'Error type 1->0', filename_list[i]
                        #     counter += 1
                        # else:
                        #     os.remove(os.path.join(FLAGS.images_dir, filename_list[i]))
                        counter += 1
                    else:
                        os.remove(os.path.join(FLAGS.images_dir, filename_list[i]))
                print 'Find: {}'.format(counter)
        elif FLAGS.eval:
            dnn.eval()
        elif FLAGS.eval_G:
            dnn.load_network()
            filename_list = [os.path.join(FLAGS.eval_G_dir, filename) for filename in os.listdir(FLAGS.eval_G_dir) if '.png' in filename]
            for filename in filename_list:
                samples = np.zeros((64, 28, 28, 1))
                im = Image.open(filename)
                im_data = np.array(im) / 255.0
                for i in range(8):
                    for j in range(8):
                        samples[i*8+j, :] = im_data[i*28: (i+1)*28, j*28: (j+1)*28].reshape([28, 28, 1])
                predict = dnn.predict(samples)
                a, b = filename.split('.')[0].split('_')[-2:]
                counter = int(a) * 859 + int(b)
                print '{}: {}'.format(counter, (predict == 5).sum() / float(predict.shape[0]))


if __name__ == '__main__':
    """
    Fit:        python run_dnn.py --train
    Predict:    python run_dnn.py --predict
    Eval:       python run_dnn.py --eval
    Eval G:     python run_dnn.py --eval_G
    Retrain:    python run_dnn.py --retrain
    """
    tf.app.run()
