from DualNumber.DualArithmetic import DualNumber, DualGrad
from DualNumber import TensorDual as td

import tensorflow as tf
import pdb

@tf.function
def mae( targetReal, targetInft, predReal, predInft ):
    diff = td.subtract( targetReal, targetInft, predReal, predInft )
    return td.abs( *diff )

#@tf.function
def mse( targetReal, targetInft, predReal, predInft ):
    diff = td.subtract( targetReal, targetInft, predReal, predInft )
    return td.power( *diff, 2.0, 0.0 )

@tf.function
def huber( targetReal, targetInft, predReal, predInft, delta=1.0 ):
    diff = td.subtract( targetReal, targetInft, predReal, predInft )
    ltd = td.multiply( *td.subtract( *diff, 0.5 * delta, 0.0 ), delta, 0.0 )
    full = td.multiply( *td.power( *diff, 2.0, 0.0 ), 0.5, 0.0 )
    return td.where( tf.greater( diff[ 0 ], delta ), *ltd, *full )

#@tf.function
def hinge( targetReal, targetInft, predReal, predInft ):
    t = tf.math.sign( targetReal )
    loss = td.subtract( 1.0, 0.0, *td.multiply( predReal, predInft, t, 0.0 ) )
    return td.where( tf.greater( loss[ 0 ], 0.0 ), *loss, 0.0, 0.0 )

@tf.function
def crossEntropy( targetReal, targetInft, predReal, predInft ):
    return td.multiply( *td.neg( targetReal, targetInft ), *td.log( predReal, predInft ) )

@tf.function
def klDivergence( targetReal, targetInft, predReal, predInft ):
    return td.multiply( targetReal, targetInft, *td.log( *td.divide( targetReal, targetInft, predReal, predInft ) ) )

@tf.function
def jsDivergence( targetReal, targetInft, predReal, predInft ):
    m = td.multiply( 0.5, 0.0, *td.add( targetReal, targetInft, predReal, predInft ) )
    tpart = td.multiply( targetReal, targetInft, *td.log( *td.divide( targetReal, targetInft, *m ) ) )
    ppart = td.multiply( predReal, predInft, *td.log( *td.divide( predReal, predInft, *m ) ) )
    return td.multiply( 0.5, 0.0, *td.add( *tpart, *ppart ) )

class Loss( object ):
    def __init__( self, func, *args ):
        self.f_ = func
        self.args_ = args
    
    def f( self, target, prediction ):
        if isinstance( prediction, DualNumber ):
            out = self.f_( target, 0.0, prediction.x_, prediction.e_, *self.args_ )
            if isinstance( prediction, DualGrad ):
                return type( prediction )( *out, prediction.n_ )
            return type( prediction )( *out )
        return self.f_( target, 0.0, prediction, 0.0, *self.args_ )[ 0 ]
    
    def df( self, target, prediction ):
        if isinstance( prediction, DualNumber ):
            prediction = prediction.x_
        return self.f_( target, 0.0, prediction, 1.0, *self.args_ )[ 1 ]