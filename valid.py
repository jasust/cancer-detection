import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from model import Model

DEBUG = True
classes = ['0', '1']
model_name = 'final'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          name='confusion'):

    if normalize:
        cm = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix [%]'
    else:
        title = 'Confusion matrix, without normalization'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('..\\checkpoint' + os.sep + model_name + os.sep + name + '.png')
    plt.close()

sess = tf.InteractiveSession()
model = Model()
model.load('..'+os.sep+'data'+os.sep+'valid_images.npy', '..'+os.sep+'data'+os.sep+'valid_labels.npy')

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('..' + os.sep + 'checkpoint' + os.sep + model_name)
saver.restore(sess, ckpt.model_checkpoint_path)

labels = model.dataset_y
predictions = np.zeros([labels.shape[0]], np.int32)
for i in range(300):
    predictions[i*200:(i+1)*200] = sess.run(model.pred, feed_dict={model.x: model.dataset_x[i*200:(i+1)*200],
                                                                   model.y: model.dataset_y[i*200:(i+1)*200]})

predictions[60000:60025] = sess.run(model.pred, feed_dict={model.x: model.dataset_x[60000:60025],
                                                           model.y: model.dataset_y[60000:60025]})
conf = sess.run(model.sm, feed_dict={model.x: model.dataset_x[60000:60025],
                                     model.y: model.dataset_y[60000:60025]})

if DEBUG:
    for i in range(25):

        plt.imshow(np.reshape(model.dataset_x[60000+i], (48, 48)), cmap="gray")
        title = ''
        if labels[60000+i]:
            title += 'Ima tumora, predikcija: '
        else:
            title += 'Nema tumora, predikcija: '
        if predictions[60000+i]:
            title += 'ima tumora '
            title += str(int(conf[i][predictions[60000+i]]*100))
            title += '%'
        else:
            title += 'nema tumora '
            title += str(int(conf[i][predictions[60000+i]]*100))
            title += '%'
        plt.title(title)
        plt.savefig('..\\checkpoint' + os.sep + model_name + os.sep + 'img_' + str(i) + '.png')
        plt.close()

    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, classes=classes)
    plot_confusion_matrix(cm, normalize=True, classes=classes, name='confusion_norm')

acc = np.sum(predictions == labels)/labels.shape[0]
print("Validation Accuracy: %.2f%%" % (acc*100))
