import os
import csv
import numpy as np
from PIL import Image

labelFolder = '..' + os.sep + 'data'
labelFile = labelFolder + os.sep + 'train_labels.csv'
testFolder = '..' + os.sep + 'data' + os.sep + 'test'
trainFolder = '..' + os.sep + 'data' + os.sep + 'train'

def process_train():

    labels = []
    images = []
    num_train = 160000
    print('Processing training data...')

    with open(labelFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)

        itr = 0
        for row in csv_reader:
            name = row[0] + '.tif'
            labels.append(int(row[1]))

            im = Image.open(trainFolder + os.sep + name).convert('L')
            im = im.resize((48, 48), Image.ANTIALIAS)
            images.append(np.array(im))

            itr += 1

            if itr % 20000 == 0:

                print('Processed', itr, 'images...')

            if itr == num_train:

                print('Writing training data...')
                np.save(labelFolder + os.sep + 'train_labels', np.array(labels))
                np.save(labelFolder + os.sep + 'train_images', np.array(images))

                labels = []
                images = []

    print('Writing validation data...')
    np.save(labelFolder + os.sep + 'valid_labels', np.array(labels))
    np.save(labelFolder + os.sep + 'valid_images', np.array(images))


def process_test():

    names = []
    images = []
    print('Processing test data...')

    for subdir, dirs, files in os.walk(testFolder):
        for fileName in files:

            name = fileName.split('.')[0]
            names.append(name)

            im = Image.open(testFolder + os.sep + fileName).convert('L')
            im = im.resize((48, 48), Image.ANTIALIAS)
            images.append(np.array(im))

    print('Writing test data...')
    np.save(labelFolder + os.sep + 'test_names', np.array(names))
    np.save(labelFolder + os.sep + 'test_images', np.array(images))


if __name__ == '__main__':

    process_train()

    process_test()
