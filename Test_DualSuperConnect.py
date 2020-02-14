from DualSuperConnect import ScnnRegressor

from DualNumber.TestLib import runTest

from sklearn.datasets.california_housing import fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np
import Update

# System test for update classes and Dual Least Squares

def test():
   # data
   cal_housing = fetch_california_housing()
   X = cal_housing.data
   Y = np.reshape( cal_housing.target, ( -1, 1 ) )

   # train models
   iters = 100
   name = [ "0", "1", "2", "3", "4" ]
   model = [ ScnnRegressor( 8, 1, 0, update=Update.Rprop() ),
             ScnnRegressor( 8, 1, 1, update=Update.Rprop() ),
             ScnnRegressor( 8, 1, 2, update=Update.Rprop() ),
             ScnnRegressor( 8, 1, 3, update=Update.Rprop() ),
             ScnnRegressor( 8, 1, 4, update=Update.Rprop() ) ]
   error = np.zeros( ( len( model ), iters ) )
   for i in range( iters ):
      for m in range( len( model ) ):
         error[ m, i ] = model[ m ].partial_fit( X, Y )
      print( i, "complete" )
   
   # plot results
   plt.figure()
   plt.title( 'Error Curves' )
   for m in range( len( model ) ):
      plt.semilogy( error[ m ], label=name[ m ] )
   plt.legend()

   plt.show()

if __name__ == "__main__":
   runTest( test )