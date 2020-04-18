# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:08:06 2020

@author: 潘慧杰
"""
import tensorflow as tf

def gram_matrix(x):
        z = tf.reshape(x, [-1, tf.shape(x)[-1]])  # this makes z [H*W, C]
        z = tf.matmul(tf.transpose(z), z) / tf.cast(tf.shape(z)[0], dtype=tf.float32)
        return z


def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                         for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics,
          end=end)