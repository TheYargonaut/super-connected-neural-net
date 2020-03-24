from DualNumber import TensorDual as td

import tensorflow as tf

#@tf.function
def makeInft( shape, total, first ):
    inft = tf.eye( total, dtype=tf.float32 )
    inft = inft[ :, first : first + tf.reduce_prod( shape ) ]
    return tf.reshape( inft, ( total, *shape ) )

#@tf.function
def zero( shape, total, first=0 ):
    real = tf.zeros( shape, dtype=tf.float32 )
    inft = makeInft( real.shape, total, first )
    return real, inft

@tf.function
def normal( shape, total, first=0, mean=0, std=1 ):
    '''randomize initial weights'''
    real = tf.random.normal( shape, mean=mean, stddev=std, dtype=tf.float32 )
    inft = makeInft( real.shape, total, first )
    return real, inft

@tf.function
def he( shape, total, first=0, n=None ):
    '''for relu.
    n should be cardinality of parameters in layer or of sum of inputs and outputs for layer'''
    if not n:
        n = total
    std = tf.sqrt( tf.divide( 2, n ) )
    return normal( shape, total, first, std=std )

@tf.function
def xavier( shape, total, first=0, n=None ):
    '''for tanh.
    n should be cardinality of parameters in layer or of sum of inputs and outputs for layer'''
    if not n:
        n = total
    std = tf.sqrt( tf.divide( 1, n ) )
    return normal( shape, total, first, std=std )