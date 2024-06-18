import os
import csv
import numpy as np
import tensorflow as tf
from model import Model

model_name = 'final'
file_name = '..' + os.sep + 'checkpoint' + os.sep + model_name + os.sep + 'submission_file.csv'

sess = tf.InteractiveSession()
model = Model()
model.load('..'+os.sep+'data'+os.sep+'test_images.npy')
names = np.load('..'+os.sep+'data'+os.sep+'test_names.npy')

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('..' + os.sep + 'checkpoint' + os.sep + model_name)
saver.restore(sess, ckpt.model_checkpoint_path)

predictions = np.zeros([names.shape[0]], np.int32)
for i in range(287):
    predictions[i*200:(i+1)*200] = sess.run(model.pred, feed_dict={model.x: model.dataset_x[i*200:(i+1)*200]})
predictions[57400:] = sess.run(model.pred, feed_dict={model.x: model.dataset_x[57400:]})

with open(file_name, mode='w', newline='') as submission_file:
    submission_writer = csv.writer(submission_file, delimiter=',')
    submission_writer.writerow(['id', 'label'])

    for i in range(predictions.shape[0]):
        submission_writer.writerow([names[i], str(predictions[i])])
