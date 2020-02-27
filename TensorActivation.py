from DualNumber.DualArithmetic import DualNumber, DualGrad
from DualNumber import TensorDual as td

import tensorflow as tf
import pdb

@tf.function
def identity( real, inft ):
    return real, inft

@tf.function
def softplus( real, inft ):
    return td.log( *td.add( *td.exp( real, inft ), 1.0, 0.0 ) )

@tf.function
def relu( real, inft ):
    return td.where( tf.greater( real, 0.0 ), real, inft, 0.0, 0.0 )

@tf.function
def leaky( real, inft, p=0.01 ):
    low = td.multiply( real, inft, p, 0.0 )
    return td.where( tf.greater( real, 0.0 ), real, inft, *low )

#@tf.function # breaks the function right now
def elu( real, inft ):
    low = td.subtract( *td.exp( real, inft ), 1.0, 0.0 )
    return td.where( tf.greater( real, 0.0 ), real, inft, *low )

@tf.function
def logistic( real, inft ):
    return td.divide( 1.0, 0.0, *td.add( 1.0, 0.0, *td.exp( -real, -inft ) ) )

@tf.function
def tanh( real, inft ):
    return td.tanh( real, inft )

@tf.function
def softmax( real, inft, t=1 ):
    raw = td.exp( *td.divide( real, inft, t, 0.0 ) )
    cs = td.sum( *raw, -1 )
    return td.divide( *raw, *cs )

# utility wrapper class

class Activation( object ):
    def __init__( self, func, *args ):
        self.f_ = func
        self.args_ = args
    
    def f( self, value ):
        if isinstance( value, DualNumber ):
            out = self.f_( tf.constant( value.x_ ), tf.constant( value.e_ ), *self.args_ )
            if isinstance( value, DualGrad ):
                return type( value )( *out, value.n_ )
            return type( value )( *out )
        return self.f_( tf.constant( value ), tf.constant( 0.0 ), *self.args_ )[ 0 ]
    
    def df( self, value ):
        if isinstance( value, DualNumber ):
            value = value.x_
        return self.f_( tf.constant( value ), tf.constant( 1.0 ), *self.args_ )[ 1 ]