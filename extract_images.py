from ops import *
import input_data
import os

mnist = input_data.read_data_sets('dataset', one_hot=True)

num = mnist.train.num_examples
images = mnist.train.images
labels = mnist.train.labels

if not os.path.exists('dataset/train_images'):
        os.makedirs('dataset/train_images')

for i in range(num):
    data = images[i].reshape([1, 28, 28, 1])
    filepath = 'dataset/train_images/{}_{}.jpg'.format(list(labels[i]).index(1), i)
    save_images(data, (1, 1), filepath)
