
import tensorflow as tf
import numpy as np

list_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
list_b = np.array([[1, 2], [3, 4], [5,6], [7,8], [9,9]])
list_label = list_b

dataset = tf.data.Dataset.from_tensor_slices({'a':list_a, 'b':list_b, 'label':list_label})
dataset = dataset.batch(2)

iterator = dataset.make_one_shot_iterator()

with tf.Session() as sess:
    while True:
        try:
            print(sess.run(iterator.get_next()))
        except tf.errors.OutOfRangeError:
            break
