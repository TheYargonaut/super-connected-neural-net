import tensorflow as tf
import pdb

# add settings later; currently, hardcode activation/loss/etc

class LinearModel( object ):
    def __init__( self, inputSize, outputSize ):
        self.weight = tf.zeros( ( inputSize, outputSize ), dtype=tf.float32 )
        self.nParams = tf.size( self.weight )
        self.inft = tf.reshape( tf.eye( self.nParams, dtype=tf.float32 ), ( self.nParams, *self.weight.shape ) )
    
    def predict( self, X ):
        return tf.tensordot( tf.cast( X, dtype=tf.float32 ), self.weight, axes=[ [ 1 ], [ 0 ] ] )
    
    def loss( self, X, y ):
        return ( self.predict( X ) - y ) ** 2

    def partial_fit( self, X, y ):
        # with tf.autodiff.ForwardAccumulator( primals=self.weight, tangents=self.weight ) as acc:
        with tf.GradientTape() as g:
            g.watch( self.weight )
            loss = tf.reduce_mean( self.loss( X, y ) )
        grad = g.gradient( loss, self.weight )
        # grad = acc.jvp( loss )
        # pdb.set_trace()
        self.weight -= 0.00001 * grad
        return loss