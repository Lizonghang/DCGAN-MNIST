import os
import input_data
from ops import *


class DCGAN(object):
    def __init__(self, sess, lr, beta1, epoch, batch_size, train_size,
                 z_dim, dataset, checkpoint_dir, logs_dir, images_dir,
                 gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024):
        self.sess = sess
        self.lr = lr
        self.beta1 = beta1
        self.epoch = epoch
        self.batch_size = batch_size
        self.train_size = train_size
        self.z_dim = z_dim
        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.images_dir = images_dir

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.build_model()

    def build_model(self):
        self.y = tf.placeholder(tf.float32, [self.batch_size, 10], name='y')
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, 28, 28, 1], name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        inputs = self.inputs

        self.G = self.generator(self.z, self.y)

        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)

        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        self.sampler_op = self.sampler(self.z, self.y)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D))
        )
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_))
        )
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))
        )

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def train(self):
        d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)

        mnist = input_data.read_data_sets('dataset', one_hot=True)

        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
        summary_op = tf.summary.merge_all()

        counter = 1
        batch_idxs = min(mnist.train.num_examples, self.train_size) / self.batch_size
        for epoch in xrange(self.epoch):
            for idx in xrange(0, batch_idxs):
                batch_images, batch_labels = mnist.train.next_batch(self.batch_size)
                batch_images = np.reshape(batch_images, [self.batch_size, 28, 28, 1])
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                self.sess.run(d_optim, feed_dict={
                    self.inputs: batch_images,
                    self.z: batch_z,
                    self.y: batch_labels
                })

                # Update G network
                self.sess.run(g_optim, feed_dict={
                    self.z: batch_z,
                    self.y: batch_labels
                })

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                self.sess.run(g_optim, feed_dict={
                    self.z: batch_z,
                    self.y: batch_labels
                })

                errD_fake = self.d_loss_fake.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })
                errD_real = self.d_loss_real.eval({
                    self.inputs: batch_images,
                    self.y: batch_labels
                })
                errG = self.g_loss.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })

                counter += 1
                print("Epoch: [%2d] [%4d/%4d], d_loss: %.8f, g_loss: %.8f" %
                      (epoch, idx, batch_idxs, errD_fake + errD_real, errG))

                if np.mod(counter, 1) == 0:
                    sample_z = np.random.uniform(-1, 1, [64, self.z_dim]).astype(np.float32)
                    sample_labels = np.ones([self.batch_size, 1]) * np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
                    samples = self.sess.run(self.sampler_op, feed_dict={
                        self.z: sample_z,
                        self.y: sample_labels,
                    })
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(self.images_dir, epoch, idx))

                    self.writer.add_summary(self.sess.run(summary_op, feed_dict={
                        self.inputs: batch_images,
                        self.y: batch_labels,
                        self.z: batch_z
                    }), epoch)

            self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), global_step=epoch)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, 10])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(conv2d(x, 11, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + 10, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = tf.concat([h1, y], 1)

            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = tf.concat([h2, y], 1)

            h3 = linear(h2, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            yb = tf.reshape(y, [self.batch_size, 1, 1, 10])
            z = tf.concat([z, y], 1)

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * 7 * 7, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, 7, 7, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, 14, 14, self.gf_dim * 2], name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, 28, 28, 1], name='g_h3'))

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, 10])
            z = tf.concat([z, y], 1)

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
            h0 = tf.concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * 7 * 7, 'g_h1_lin'), train=False))
            h1 = tf.reshape(h1, [self.batch_size, 7, 7, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(
                self.g_bn2(deconv2d(h1, [self.batch_size, 14, 14, self.gf_dim * 2], name='g_h2'), train=False))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, 28, 28, 1], name='g_h3'))

    def generate(self, gen_y):
        saver = tf.train.Saver(tf.global_variables())

        if not self.load_checkpoint(saver):
            raise Exception("[ERROR] No checkpoint file found!")

        z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        y = np.zeros((self.batch_size, 10))
        y[:, gen_y] = np.ones(self.batch_size)

        return self.sess.run(self.G, feed_dict={self.z: z, self.y: y})

    def load_checkpoint(self, checkpoint_dir):
        import re
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            return True, counter
        else:
            return False, 0
