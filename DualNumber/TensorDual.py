try:
    from DualArithmetic import DualNumber, DualGrad
except:
    from DualNumber.DualArithmetic import DualNumber, DualGrad
import tensorflow as tf

# Handle dual numbers with Tensorflow. This takes a functional approach to better acommodate low-level tensorflow
# Includes wrapper class equivalent to those in DualArithmetic

# Unary operators

@tf.function
def log( xReal, xInft ):
    real = tf.math.log( xReal )
    inft = tf.math.divide( xInft, xReal )
    return real, inft

@tf.function
def exp( xReal, xInft ):
    real = tf.math.exp( xReal )
    inft = tf.math.multiply( real, xInft )
    return real, inft

@tf.function
def tanh( xReal, xInft ):
    real = tf.math.tanh( xReal )
    inft = tf.math.multiply( xInft, tf.math.subtract( 1.0, tf.math.pow( real, 2.0 ) ) )
    return real, inft

@tf.function
def abs( xReal, xInft ):
    real = tf.math.abs( xReal )
    inft = tf.where( xReal > 0, xInft, -xInft )
    return real, inft

@tf.function
def neg( xReal, xInft ):
    real = tf.math.negative( xReal )
    inft = tf.math.negative( xInft )
    return real, inft

# Binary Operators

@tf.function
def add( xReal, xInft, yReal, yInft ):
    real = tf.math.add( xReal, yReal )
    inft = tf.math.add( xInft, yInft )
    return real, inft

@tf.function
def subtract( xReal, xInft, yReal, yInft ):
    real = tf.math.subtract( xReal, yReal )
    inft = tf.math.subtract( xInft, yInft )
    return real, inft

@tf.function
def multiply( xReal, xInft, yReal, yInft ):
    real = tf.math.multiply_no_nan( xReal, yReal )
    inft = tf.math.add( tf.math.multiply_no_nan( xReal, yInft ), tf.math.multiply_no_nan( xInft, yReal ) )
    return real, inft

@tf.function
def divide( xReal, xInft, yReal, yInft ):
    real = tf.math.divide( xReal, yReal )
    inft = tf.math.divide( ( tf.math.multiply( xInft, yReal ) -
                             tf.math.multiply( xReal, yInft ) ),
                           tf.math.pow( yReal, 2 ) )
    return real, inft

@tf.function
def power( xReal, xInft, yReal, yInft ):
    real = tf.math.pow( xReal, yReal )
    if yInft:
        inft = tf.math.multiply_no_nan(
        real,
        tf.math.add(
            tf.math.divide(
                tf.math.multiply_no_nan( yReal, xInft ),
                xReal ),
            tf.math.multiply_no_nan(
                yInft,
                tf.math.log( xReal ) ) ) )
        return real, inft
    inft = tf.math.multiply_no_nan( tf.math.pow( xReal, tf.math.subtract( yReal, 1.0 ) ), tf.math.multiply_no_nan( xInft, yReal ) )
    return real, inft

def tensordot( xReal, xInft, yReal, yInft, axes=[ [ -1 ], [ 0 ] ] ):
    real = tf.tensordot( xReal, yReal, axes=axes )
    ie = [ [ ( a if a < 0 else a + 1 ) for a in t ] for t in axes ]
    inft = tf.tensordot( yInft, xReal, axes=[ ie[ 1 ], axes[ 0 ] ] ) + tf.tensordot( xInft, yReal, axes=[ ie[ 0 ], axes[ 1 ] ] )
    return real, inft

# utilities

@tf.function
def where( condition, xReal, xInft, yReal, yInft ):
    real = tf.where( condition, xReal, yReal )
    inft = tf.where( condition, xInft, yInft )
    return real, inft

@tf.function
def sum( xReal, xInft, axis=0, keepdims=False ):
    eAxis = axis if axis < 0 else axis + 1
    real = tf.math.reduce_sum( xReal, axis, keepdims )
    inft = tf.math.reduce_sum( xInft, eAxis, keepdims )
    return real, inft

# Wrapper class

class DualTensor( DualGrad ):
    '''Dual numbers using tensorflow tensors to represent multiple
    independant variables with dual numbers'''
    
    # binary ops
    def __add__( self, other ):
        if isinstance( other, DualNumber ):
            return DualTensor( *add( self.x_, self.e_, other.x_, other.e_ ), self.n_ )
        return DualTensor( *add( self.x_, self.e_, other, 0.0 ), self.n_ )
    def __sub__( self, other ):
        if isinstance( other, DualNumber ):
            return DualTensor( *subtract( self.x_, self.e_, other.x_, other.e_ ), self.n_ )
        return DualTensor( *subtract( self.x_, self.e_, other, 0.0 ), self.n_ )
    def __rsub__( self, other ):
        if isinstance( other, DualNumber ):
            return DualTensor( *subtract( other.x_, other.e_, self.x_, self.e_ ), self.n_ )
        return DualTensor( *subtract( other, 0.0, self.x_, self.e_ ), self.n_ )
    def __mul__( self, other ):
        if isinstance( other, DualNumber ):
            return DualTensor( *multiply( self.x_, self.e_, other.x_, other.e_ ), self.n_ )
        return DualTensor( *multiply( self.x_, self.e_, other, 0.0 ), self.n_ )
    def __truediv__( self, other ):
        if isinstance( other, DualNumber ):
            return DualTensor( *divide( self.x_, self.e_, other.x_, other.e_ ), self.n_ )
        return DualTensor( *divide( self.x_, self.e_, other, 0.0 ), self.n_ )
    def __rtruediv__( self, other ):
        if isinstance( other, DualNumber ):
            return DualTensor( *divide( other.x_, other.e_, self.x_, self.e_ ), self.n_ )
        return DualTensor( *divide( other, 0.0, self.x_, self.e_ ), self.n_ )
    def __pow__( self, other ):
        if isinstance( other, DualNumber ):
            return DualTensor( *power( self.x_, self.e_, other.x_, other.e_ ), self.n_ )
        return DualTensor( *power( self.x_, self.e_, other, 0.0 ), self.n_ )
