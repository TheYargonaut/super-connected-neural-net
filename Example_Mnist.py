import Activation, Error, Update
from DualNumber.TestLib import runTest
from DualLeastSquares import LLS
from DualSuperConnect import SCNN
from SkModel import formatClassTarget as fct

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

import pdb

# Data
n_components = 100
train_samples = 500 # max 70000; Memory error at 5000, need to do some batching to make it work
X, y = fetch_openml( 'mnist_784', version=1, return_X_y=True )
print( 'Data Download Complete' )
random_state = check_random_state( 0 )
permutation = random_state.permutation( X.shape[ 0 ] )
X = X[ permutation ]
y = y[ permutation ]
X = X.reshape( ( X.shape[ 0 ], -1 ) )
X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=train_samples, test_size=10000 )
pca = PCA( n_components=n_components )
X_train = pca.fit_transform( X_train )
X_test = pca.transform( X_test )
Y_train, Y_test = fct( y_train ), fct( y_test )

def linearMnist():
   model = LLS( n_components, Y_train.shape[ 1 ],
                outputAct=Activation.Identity(),
                update=Update.Rprop(),
                error=Error.Hinge() )
   iters = 10
   error = np.zeros( iters )
   for i in range( iters ):
      error[ i ] = model.partial_fit( X_train, Y_train )
      print( i + 1, 'complete' )
   
   # Plot training results
   plt.figure()
   plt.title( 'Error Curve' )
   plt.semilogy( error )
   plt.show()

   # TODO: accuracy on training and test sets
   # TODO: fix the stupid invalid values on softmax + JS

   pdb.set_trace()

def scnnMnist():
   pass

if __name__  == "__main__":
   runTest( linearMnist )
   # runTest( scnnMnist )