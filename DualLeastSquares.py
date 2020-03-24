import Update, Error, Regularize, Activation, Initialize
from DualNumber.DualArithmetic import DualGrad as Dual
from Train import batch

import numpy as np
import pdb

class LLS( object ):
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
   
   def partial_fit( self, X, Y, maxBatch=None ):
      '''Perform single round of training without reset. Returns average error.'''
      grad, loss = np.zeros_like( self.weight_.x_ ), 0
      for bx, by in batch( X, Y, maxBatch ):
         portion = float( len( bx ) ) / len( X )
         bgrad, bloss = self.trainGrad_( bx, by )
         grad += bgrad * portion
         loss += bloss * portion
      grad += self.regularizeGrad_()
      self.weight_ = self.update_( self.weight_, grad )
      return loss

   def predict( self, X, maxBatch=None ):
      '''X should be 2d numpy array'''
      if maxBatch is None:
         maxBatch = len( X )
      if X.shape[ 1 ] == self.inputSize_ - 1:
         X = np.append( np.ones( ( X.shape[ 0 ], 1 ) ), X, 1 )
      
      yPred = Dual( np.zeros( ( len( X ), self.outputSize_ ) ), 0, self.weight_.n_ )
      lower = 0
      for bx in batch( X, maxBatch=maxBatch ):
         raw = Dual( bx, n=self.nParameters_ ).matmul( self.weight_ )
         upper = lower + len( bx )
         yPred[ lower:upper ] = self.outputActivation_( raw )
         lower += maxBatch
      return yPred
   
   def error( self, X, Y ):
      '''X and y should both be 2d numpy arrays'''
      return self.errorFunc_( Y, self.predict( X ) )

   def regularizeGrad_( self ):
      '''return gradient for regularization of weights in same shape as weights'''
      grad = self.regLoss_( self.weight_ ).e_
      while len( grad.shape ) > 1:
         grad = np.sum( grad, 1 )
      return grad.reshape( self.weight_.x_.shape )
   
   def trainGrad_( self, X, Y ):
      '''return gradient for training weights in same shape as weights as well as the average loss'''
      loss = self.error( X, Y )
      grad = loss.e_
      grad = np.average( grad, 1 ) # across samples
      while len( grad.shape ) > 1:
         grad = np.sum( grad, 1 ) # across outputs
      return grad.reshape( self.weight_.x_.shape ), np.average( loss.x_ )

   def cool( self ):
      self.updateObj_.cool()