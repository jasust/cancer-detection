import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import slim


class Model:

    def __init__(self, training=False):

        self.height = 48
        self.width = 48
        self.num_classes = 2
        self.folder = 'final'
        if not os.path.exists(path='..' + os.sep + 'checkpoint' + os.sep + self.folder):
            os.mkdir(path='..' + os.sep + 'checkpoint' + os.sep + self.folder)

        self.num_epochs = 24
        self.batch_size = 200
        self.display_step = 40
        self.point_step = 20
        self.save_step = 2

        self.loss_f = []
        self.acc_f = []

        def conv_net(data, is_training):

            with slim.arg_scope([slim.conv2d], padding='SAME', normalizer_fn=slim.batch_norm,
                                normalizer_params={'decay': 0.9997, 'is_training': True, 'updates_collections': None,
                                                   'trainable': is_training},
                                weights_initializer=slim.initializers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.005)):

                conv1 = slim.conv2d(data, 32, [3, 3], scope='conv1')
                pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
                conv2 = slim.conv2d(pool1, 64, [5, 5], scope='conv2')
                pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
                conv3 = slim.conv2d(pool2, 128, [3, 3], scope='conv3')
                pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
                conv4 = slim.conv2d(pool3, 256, [3, 3], scope='conv4')
                pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')

                fc1 = slim.fully_connected(tf.contrib.layers.flatten(pool4), 48, scope='fc1')
                output = slim.fully_connected(fc1, 2, activation_fn=tf.nn.sigmoid, scope='output')
                # TODO: add dropout if needed

            return output

        self.x = tf.placeholder("float32", [None, self.height, self.width, 1])
        self.y = tf.placeholder("int32", [None])

        self.logits = conv_net(self.x, is_training=training)
        self.pred = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)
        self.sm = tf.nn.softmax(self.logits)
        self.acc = tf.count_nonzero(tf.equal(self.pred, self.y))
        self.acc = self.acc / self.batch_size

        # class_weight = tf.constant([1.3, 0.7])
        # self.weighted_logits = tf.multiply(self.logits, class_weight)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)

        self.loss = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=5e-4, use_locking=False).minimize(self.loss)

        print("Initialization finished...")

    def load(self, set_name, labels_name=''):

        data = np.load(set_name)
        # data1 = np.load('..'+os.sep+'data'+os.sep+'valid_images.npy')
        # data = np.vstack((data, data1))
        data = np.reshape(data, (-1, 48, 48, 1))
        self.dataset_x = data.astype(np.float32)*(1./255)-0.5

        if not (labels_name == ''):
            labels = np.load(labels_name)
            # labels1 = np.load('..' + os.sep + 'data' + os.sep + 'valid_labels.npy')
            # labels = np.concatenate((labels, labels1))
            self.dataset_y = labels.astype(np.int32)

        # TODO: generate more data by mirroring here if needed
        print("Dataset loaded...")

    def train(self):

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for i in range(self.num_epochs):

                step = 1
                loss = 0
                acc = 0
                start_time = time.time()

                while step * self.batch_size <= len(self.dataset_x):

                    batch_x = self.dataset_x[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_y = self.dataset_y[(step - 1) * self.batch_size:step * self.batch_size]

                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})

                    if step % self.point_step == 0:

                        loss_value, acc_value = sess.run([self.loss, self.acc], feed_dict={self.x: batch_x, self.y: batch_y})
                        loss += loss_value
                        acc += acc_value

                        if step % self.display_step == 0:
                            print("Epoch %d, step %d: loss = %f, acc = %.2f%%" % (i+1, step, loss_value, 100*acc_value))

                    step += 1

                self.loss_f.append(loss*self.point_step/(step-1))
                self.acc_f.append(100*acc*self.point_step/(step-1))

                duration = time.time() - start_time
                print("Duration of training for this epoch: %.2fs" % duration)
                if (i+1) % self.save_step == 0:
                    saver.save(sess, '..\\checkpoint' + os.sep + self.folder + os.sep + 'model.ckpt', i+1)

            print("Optimization Finished...")

        plt.plot(self.acc_f)
        plt.title("Accuracy in %")
        plt.savefig('..\\checkpoint' + os.sep + self.folder + os.sep + 'acc.png')
        plt.close()
        plt.plot(self.loss_f)
        plt.title("Loss function")
        plt.savefig('..\\checkpoint' + os.sep + self.folder + os.sep + 'loss.png')
        plt.close()

if __name__ == '__main__':

    model = Model(training=True)
    model.load('..'+os.sep+'data'+os.sep+'train_images.npy', '..'+os.sep+'data'+os.sep+'train_labels.npy')
    model.train()
