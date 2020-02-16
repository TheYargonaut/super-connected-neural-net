import Activation, Error, Update
from DataFetch import fetch
from DualNumber.TestLib import runTest
from DualLeastSquares import LLS
from DualSuperConnect import SCNN
from SkModel import formatClassTarget as fct

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import pdb

np.seterr(invalid='raise')

# Data
n_components = 100
train_samples = 5000 # max 70000; Memory error at 5000, need to do some batching to make it work
X, y = fetch( 'mnist', lambda : fetch_openml( 'mnist_784', version=1, return_X_y=True ) )
print( 'Data Fetch Complete' )
random_state = check_random_state( 0 )
permutation = random_state.permutation( X.shape[ 0 ] )
X = X[ permutation ]
y = y[ permutation ]
X = X.reshape( ( X.shape[ 0 ], -1 ) )
X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=train_samples, test_size=10000 )
y_train, y_test = y_train.astype( int ), y_test.astype( int )

# Preprocessing
pca = PCA( n_components=n_components )
scaler = StandardScaler()
X_train = scaler.fit_transform( pca.fit_transform( X_train ) )
X_test = scaler.transform( pca.transform( X_test ) )
Y_train, Y_test = fct( y_train ), fct( y_test )

def linearMnist():
   model = LLS( n_components, Y_train.shape[ 1 ],
                outputAct=Activation.Softmax(),
                update=Update.Rprop(),
                error=Error.JsDivergence() )
   iters = 10
   loss = np.zeros( iters )
   train_acc = np.zeros( iters + 1 )
   test_acc = np.zeros( iters + 1 )
   for i in range( iters ):
      raw_pred = model.predict( X_train ).x_
      train_acc[ i ] = np.average( np.argmax( raw_pred, axis=1 ) == y_train )
      raw_pred = model.predict( X_test ).x_
      test_acc[ i ] = np.average( np.argmax( raw_pred, axis=1 ) == y_test )
      loss[ i ] = model.partial_fit( X_train, Y_train )
      print( i + 1, 'complete' )
   raw_pred = model.predict( X_train ).x_
   train_acc[ -1 ] = np.average( np.argmax( raw_pred, axis=1 ) == y_train )
   raw_pred = model.predict( X_test ).x_
   test_acc[ -1 ] = np.average( np.argmax( raw_pred, axis=1 ) == y_test )
   
   # Plot training results
   plt.figure()
   plt.title( 'Loss Curve' )
   plt.semilogy( loss )
   plt.figure()

   plt.title( 'Accuracy Curves' )
   plt.semilogy( train_acc, label='train' )
   plt.semilogy( test_acc, lable='test')
   plt.legend()
   plt.show()

   pdb.set_trace()

def scnnMnist():
   pass

if __name__  == "__main__":
   runTest( linearMnist )
   # runTest( scnnMnist )