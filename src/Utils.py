import datetime

import numpy as np
import tensorflow as tf
import scipy.signal
from helper import *
#from vizdoom import *

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
    #s = frame[10:-10,30:-30]
    #s = scipy.misc.imresize(s,[84,84])
    #s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return frame

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def now(dateformat='%Y%m%d'):
    return datetime.datetime.today().strftime(dateformat)

def variable_summaries(writer,summary):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(summary)
        tf.summary.scalar('mean', summary)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(summary - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(summary))
        tf.summary.scalar('min', tf.reduce_min(summary))
        tf.summary.histogram('histogram', summary)
        merged = tf.summary.merge_all()
        writer.add_summary(merged)

def variable_summaries2(summary,var):
    with tf.name_scope('summaries'):
        mean = np.mean(var);
        max = np.max(var)
        min = np.min(var)
        with tf.name_scope('stddev'):
            stddev = np.sqrt(np.mean(np.square(var - mean)))

        summary.value.add(tag='mean', simple_value=float(mean))
        summary.value.add(tag='stddev', simple_value=float(stddev))
        summary.value.add(tag='max', simple_value=max)
        summary.value.add(tag='min', simple_value=min)

        #tf.summary.histogram('histogram', summary)
        #merged = tf.summary.merge_all()
        #writer.add_summary(merged)
