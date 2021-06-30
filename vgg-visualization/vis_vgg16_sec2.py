import numpy as np
import tensorflow as tf
import vgg16
import utils
import cv2


np_load_old = np.load

np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

img = utils.load_image("./test_data/tiger.jpeg")

batch = img.reshape((1, 224, 224, 3))

# with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=(
# tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.compat.v1.Session() as sess:
        images = tf.compat.v1.placeholder("float", [1, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        with tf.compat.v1.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        print("Top 5 object gan giong voi hinh tiger.jpeg:")
        utils.print_prob(prob[0], './synset.txt')
