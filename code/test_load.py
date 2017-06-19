import tensorflow as tf
import numpy as np

sess = tf.Session()
loaded = np.load('../small_bitmaps/1002/0_320x180.npy')
png = tf.image.encode_png(loaded)
pic = sess.run(png)
with open('test.png', 'wb') as output:
    output.write(pic)
