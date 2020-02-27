from DualNumber.DualArithmetic import DualNumber, DualGrad
from DualNumber import TensorDual as td

import tensorflow as tf
import pdb

@tf.function
def zero( real, inft ):
    return 0.0, 0.0

@tf.function
def ridge( real, inft, l2=1e-4 ):
    return td.multiply( *td.power( real, inft, 2.0, 0.0 ), l2 / tf.reduce_prod( tf.cast( real.shape, tf.float32 ) ), 0.0 )

@tf.function
def lasso( real, inft, l1=1e-4 ):
    return td.multiply( *td.abs( real, inft ), l1 / tf.reduce_prod( tf.cast( real.shape, tf.float32 ) ), 0.0 )

@tf.function
def elasticNet( real, inft, l1=1e-4, l2=1e-4 ):
    return td.add( *ridge( real, inft, l2 ), *lasso( real, inft, l1 ) )

# utility wrapper class

class Regularizer( object ):
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