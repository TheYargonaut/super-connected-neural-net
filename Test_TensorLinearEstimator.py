from TensorLinearEstimator import TLE
import TensorUpdate as tu
from DualNumber.TestLib import runTest

import Update

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# System test for update classes and Dual Least Squares

def test():
   # base data
   X = tf.random.normal( shape=[1000, 1], mean=50, stddev=10, dtype=tf.float32 )
   Y = X * 2 - 10
   zinft = tf.zeros( [2,1,1], dtype=tf.float32 )

   # add noise
   X += tf.random.normal( shape=[1000, 1], mean=0, stddev=2, dtype=tf.float32 )
   Y += tf.random.normal( shape=[1000, 1], mean=0, stddev=2, dtype=tf.float32 )

   # split
   trainX = X[ :900 ]
   trainY = Y[ :900 ]
   testX = X[ 900: ]
   testY = Y[ 900: ]

   # for prediction line
   plotX = tf.constant( [ [ tf.math.reduce_min( X ).numpy() ], [ tf.math.reduce_max( X ).numpy() ] ] )
   
   iters = 2000
   name = [ "Simple TF" ]
   model = [ TLE( 1, 1, update=tu.Rprop() ) ]
   error = np.zeros( ( len( model ), iters ) )
   for i in range( iters ):
      for m in range( len( model ) ):
         pf = model[ m ].partial_fit( trainX, zinft, trainY, zinft )
         error[ m, i ] = pf[ 0 ].numpy()
      print( i + 1, "complete" )

   # plot results
   plt.figure()
   plt.title( 'Data Space' )
   plt.scatter( trainX, trainY, label='train' )
   plt.scatter( testX, testY, label='test' )
   plt.plot( plotX, model[ 0 ].predict( plotX, zinft )[ 0 ].numpy(), label='prediction' )
   plt.legend()

   plt.figure()
   plt.title( 'Error Curves' )
   for m in range( len( model ) ):
      plt.semilogy( error[ m ], label=name[ m ] )
   plt.legend()

   plt.show()

   print( model[ 0 ].weight_ )
   print( model[ 0 ].bias_ )

if __name__ == "__main__":
   runTest( test )