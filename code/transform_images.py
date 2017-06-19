import tensorflow as tf
import os
import numpy as np

def decode_jpg(input_path):
    with open(input_path, 'rb') as handler:
        img = handler.read()
        bmp = tf.image.decode_jpeg(img, channels=3)
        return bmp

def resize_bmp(bmp, width, height):
    resized = tf.image.resize_images(bmp, [height, width])
    return resized

base_input_dir = os.path.join('..', 'raw_data')
base_output_dir = os.path.join('..', 'small_bitmaps')

for thing in os.listdir(base_input_dir)[100:]:
    thingpath = os.path.join(base_input_dir, thing)
    if os.path.isdir(thingpath):
        appid = thing
        print(appid)
        dirpath = os.path.join(base_input_dir, appid)
        for thing2 in os.listdir(dirpath):
            if thing2.endswith('.jpg'):
                image_path = os.path.join(dirpath, thing2)
                bmp = decode_jpg(image_path)
                sizes = [(320, 240), (320, 180)]
                sess = tf.Session()
                resizeds = [sess.run(resize_bmp(bmp, x, y)) for (x,y) in sizes]
                tf.reset_default_graph()
                output_dirpath = os.path.join(base_output_dir, appid)
                if not os.path.exists(output_dirpath):
                    os.mkdir(output_dirpath)
                names = [thing2[:-4] + '_' + str(x) + 'x' + str(y) + '.npy' for (x,y) in sizes]
                filepaths = [os.path.join(output_dirpath, name) for name in names]
                [np.save(filepath, r.astype('uint8')) for (r, filepath) in zip(resizeds,filepaths)]
