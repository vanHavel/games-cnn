import numpy as np
import tensorflow as tf

def bp_mll_loss(y_true, y_pred):
    stacked = tf.stack([y_true, y_pred], axis=1)
    return tf.reduce_sum(
        tf.map_fn(
            single_mll_loss,
            stacked,
            dtype='float32'
        )
    )

def single_mll_loss(row):
    y_true = row[0]
    y_pred = row[1]
    shape = tf.shape(y_true)
    size = shape[0]
    y_i = tf.equal(y_true, tf.ones(shape))
    y_i_bar = tf.not_equal(y_true, tf.ones(shape))
    truth_matrix = pairwise_and(y_i, y_i_bar)
    sub_matrix = pairwise_sub(y_pred, y_pred)
    vals = tf.gather_nd(sub_matrix, tf.where(truth_matrix))
    y_i_size = tf.shape(tf.where(y_i))[0]
    numerator = tf.reduce_sum(tf.exp(tf.negative(vals)))
    denominator = tf.to_float(y_i_size * (size - y_i_size))
    return numerator / denominator

def pairwise_sub(a, b):
    column = tf.expand_dims(a, -1)
    row = tf.expand_dims(b, 0)
    return tf.subtract(column, row)

def pairwise_and(a, b):
    column = tf.expand_dims(a, -1)
    row = tf.expand_dims(b, 0)
    return tf.logical_and(column, row)
