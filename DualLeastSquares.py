import Update, Error, Regularize, Activation, Initialize
from DualNumber.DualArithmetic import DualGrad as Dual

import numpy as np
import pdb

# Let more general 'training' class handle batches, iterations, early stopping, cooling update

class LlsRegressor( object ):
   '''Linear Least-Squares estimator using gradient descent with dual numbers'''
   _estimator_type = "regressor"

   def __init__( self, inputSize, outputSize,
                 outputAct=Activation.Identity(),
                 iweight=Initialize.zero,
                 update=Update.Momentum(),
                 error=Error.Mse(),
                 regularization=Regularize.Zero() ):
      # structure settings
      self.inputSize_ = inputSize + 1
      self.outputSize_ = outputSize

      self.nParameters_ = self.inputSize_ * self.outputSize_

      self.outputActivationObj_ = outputAct
      self.outputActivation_ = self.outputActivationObj_.f

      # training settings
      self.iweight_ = iweight
      self.errorObj_ = error
      self.errorFunc_ = error.f
      self.updateObj_ = update
      self.update_ = update.step
      self.regObj_ = regularization
      self.regLoss_ = self.regObj_.f

      # put in clean state
      self.initialize_()

   def reset( self ):
      self.updateObj_.reset()
      self.initialize_()
   
   def initialize_( self ):
      self.weight_ = self.iweight_( ( self.inputSize_, self.outputSize_ ), self.nParameters_ )

   def partial_fit( self, X, Y ):
      '''X and Y should both be 2d numpy arrays'''
      error = self.error( X, Y )
      grad = np.average( error.e_, 1 ).reshape( self.weight_.x_.shape )
      self.weight_ = self.update_( self.weight_, grad )
      return np.average( error.x_ )

   def predict( self, X ):
      '''X should be 2d numpy array'''
      if X.shape[ 1 ] == self.inputSize_ - 1:
         X = np.append( np.ones( ( X.shape[ 0 ], 1 ) ), X, 1 )
      return Dual( X, n=self.nParameters_ ).matmul( self.weight_ )
   
   def error( self, X, Y ):
      '''X and y should both be 2d numpy arrays'''
      return self.errorFunc_( Y, self.predict( X ) )

   def cool( self ):
      self.updateObj_.cool()