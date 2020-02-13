import numpy as np

# Model Utilites

def formatTarget( y, expectWidth=None ):
   '''make sure y is a 2-d numpy array
   if expectLen is provided, use to check appropriate size'''
   y = np.array( y )
   if len( y.shape ) == 1:
      if expectWidth:
         y = np.reshape( y, ( -1, expectWidth ) )
      else:
         y = np.reshape( y, ( -1, 1 ) )
   if expectWidth:
      assert y.shape[ 1 ] == expectWidth
   assert len( y.shape ) == 2   
   return y

class Regressor( object ):
   '''abstract class for regressors'''
   _estimator_type = "regressor"