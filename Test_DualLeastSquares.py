from DualLeastSquares import LLS
from DualNumber.TestLib import runTest

import Update

import matplotlib.pyplot as plt
import numpy as np

# System test for update classes and Dual Least Squares

def test():
   # base data
   X = np.random.randn( 1000, 1 ) * 10 + 50
   Y = X * 2 - 10

   # add noise
   X += np.random.randn( 1000, 1 ) * 2
   Y += np.random.randn( 1000, 1 ) * 2

   # split
   trainX = X[ :900 ]
   trainY = Y[ :900 ]
   testX = X[ 900: ]
   testY = Y[ 900: ]

   # for prediction line
   plotX = np.array( [ min( X ), max( X ) ] )
   
   iters = 2000
   name = [ "RMSProp", "Momentum", "Nesterov", "SGD", "Rprop", "Adam" ]
   model = [ LLS( 1, 1, update=Update.RmsProp() ),
             LLS( 1, 1, update=Update.Momentum( 1e-7 ) ),
             LLS( 1, 1, update=Update.NesterovMomentum( 1e-7 ) ),
             LLS( 1, 1, update=Update.Sgd( 1e-7 ) ),
             LLS( 1, 1, update=Update.Rprop() ),
             LLS( 1, 1, update=Update.Adam() ) ]
   error = np.zeros( ( len( model ), iters ) )
   for i in range( iters ):
      for m in range( len( model ) ):
         error[ m, i ] = model[ m ].partial_fit( trainX, trainY )
      print( i + 1, "complete" )

   # plot results
   plt.figure()
   plt.title( 'Data Space' )
   plt.scatter( trainX, trainY, label='train' )
   plt.scatter( testX, testY, label='test' )
   plt.plot( plotX, model[ 4 ].predict( plotX ).x_, label='prediction' )
   plt.legend()

   plt.figure()
   plt.title( 'Error Curves' )
   for m in range( len( model ) ):
      plt.semilogy( error[ m ], label=name[ m ] )
   plt.legend()

   plt.show()

if __name__ == "__main__":
   runTest( test )