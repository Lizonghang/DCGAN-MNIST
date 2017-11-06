import os
import numpy as np
import tensorflow as tf
import input_data


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class DNN(object):

    def __init__(self, sess, lr,  batch_size, train_size, epoch, test_duration, activation_func,
                 checkpoint_dir, log_dir):
        self.sess = sess
        self.lr = lr
        self.batch_size = batch_size
        self.train_size = train_size
        self.epoch = epoch
        self.test_duration = test_duration
        self.activation_func = activation_func
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

    def inference(self, input_):
        with tf.name_scope('layer_1'):
            with tf.name_scope('conv_1'):
                W1 = weight_variable([5, 5, 1, 32])
                b1 = bias_variable([32])
                h1 = tf.nn.relu(conv2d(input_, W1) + b1)
            with tf.name_scope('pool_1'):
                p1 = max_pool_2x2(h1)

        with tf.name_scope('layer_2'):
            with tf.name_scope('conv_2'):
                W2 = weight_variable([5, 5, 32, 64])
                b2 = bias_variable([64])
                h2 = tf.nn.relu(conv2d(p1, W2) + b2)
            with tf.name_scope('pool_2'):
                p2 = max_pool_2x2(h2)

        with tf.name_scope('full_connect'):
            W_fc1 = weight_variable([7*7*64, 1024])
            b_fc1 = bias_variable([1024])
            p2_flat = tf.reshape(p2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(p2_flat, W_fc1) + b_fc1)

        with tf.name_scope('drop_out'):
            self.keep_prob = tf.placeholder('float')
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        with tf.name_scope('output_layer'):
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])
            logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return logits

    def calculate_loss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        tf.summary.scalar('loss', loss)
        return loss

    def train(self, loss, global_step):
        return tf.train.AdamOptimizer(self.lr).minimize(loss, global_step)

    def fit(self, training_with_test=True):
        mnist = input_data.read_data_sets('dataset', one_hot=True)

        train_size = min(mnist.train.num_examples, self.train_size)
        max_train_batch_idx = train_size / self.batch_size

        global_step = tf.Variable(0, trainable=False)

        self.input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='batch_input')
        self.label = tf.placeholder(tf.int32, [None, 10], name='batch_label')

        self.logits = self.inference(self.input)

        loss = self.calculate_loss(self.logits, self.label)

        train_op = self.train(loss, global_step)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        self.sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(tf.global_variables())

        for epoch in range(self.epoch):
            for batch_idx in xrange(max_train_batch_idx):
                batch_samples, batch_labels = mnist.train.next_batch(self.batch_size)
                batch_samples = np.reshape(batch_samples, [self.batch_size, 28, 28, 1])
                self.sess.run(train_op, feed_dict={
                    self.input: batch_samples,
                    self.label: batch_labels,
                    self.keep_prob: 0.7
                })

                counter = self.sess.run(global_step)
                if counter % 100 == 0:
                    writer.add_summary(self.sess.run(summary_op, feed_dict={
                        self.input: np.reshape(mnist.test.images, [mnist.test.images.shape[0], 28, 28, 1]),
                        self.label: mnist.test.labels,
                        self.keep_prob: 1.0
                    }), counter)

            saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), global_step=global_step)

            if training_with_test and np.mod(epoch, self.test_duration) == 0:
                print 'Accuracy = {0}% at epoch {1}'.format(self.sess.run(accuracy, feed_dict={
                    self.input: np.reshape(mnist.test.images, [mnist.test.images.shape[0], 28, 28, 1]),
                    self.label: mnist.test.labels,
                    self.keep_prob: 1.0
                }) * 100, epoch)

    def load_network(self):
        self.global_step = tf.Variable(0, trainable=False)

        self.inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name='batch_input')
        self.labels = tf.placeholder(tf.int32, [None, 10], name='batch_label')

        self.logits = self.inference(self.inputs)

        self.loss = self.calculate_loss(self.logits, self.labels)

        self.train_op = self.train(self.loss, self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.summary_op = tf.summary.merge_all()

        if not self.load_checkpoint(self.saver):
            raise Exception("[ERROR] No checkpoint file found!")

    def refit(self, samples_, labels_):
        self.sess.run(self.train_op, feed_dict={
            self.inputs: samples_,
            self.labels: labels_,
            self.keep_prob: 0.7
        })

        self.writer.add_summary(self.sess.run(self.summary_op, feed_dict={
            self.inputs: samples_,
            self.labels: labels_,
            self.keep_prob: 1.0
        }), self.sess.run(self.global_step))

        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'),
                        global_step=self.sess.run(self.global_step))

    def predict(self, samples):
        self.load_network()
        logits_ = self.sess.run(self.logits, feed_dict={self.inputs: samples, self.keep_prob: 1.0})
        return logits_.argmax(axis=1)

    def eval(self):
        self.load_network()
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        mnist = input_data.read_data_sets('dataset', one_hot=True)
        print 'Accuracy = {0}% in test set.'.format(self.sess.run(accuracy, feed_dict={
            self.inputs: np.reshape(mnist.test.images, [mnist.test.images.shape[0], 28, 28, 1]),
            self.labels: mnist.test.labels,
            self.keep_prob: 1.0
        }) * 100)

    def load_checkpoint(self, saver):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False
