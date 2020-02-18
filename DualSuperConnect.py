
import Update, Error, Regularize, Activation, Initialize
from DualNumber.DualArithmetic import DualGrad as Dual
from Train import batch

import numpy as np
from sklearn.metrics import mean_squared_error
import pdb

# np.seterr(invalid='raise')

# version which uses Dual Numbers for automatic forward differentiation

# Separate trainer and model?

# TODO: Elision

class SCNN( object ):
   _estimator_type = "regressor"

   def __init__( self, inputSize, outputSize, hiddenSize=100,
                 recurrent=False,
                 hiddenAct=Activation.Relu(),
                 outputAct=Activation.Identity(),
                 iweight=Initialize.he,
                 update=Update.Momentum(),
                 error=Error.Mse(),
                 regularization=Regularize.Zero() ):
      # structure settings
      self.inputSize_ = inputSize + 1
      self.outputSize_ = outputSize
      self.hiddenSize_ = hiddenSize

      self.valueInSize_ = self.inputSize_ + hiddenSize
      self.valueOutSize_ = hiddenSize + outputSize
      self.nParameters_ = self.valueInSize_ * self.valueOutSize_

      self.hiddenActivationObj_ = hiddenAct
      self.hiddenActivation_ = self.hiddenActivationObj_.f
      self.outputActivationObj_ = outputAct
      self.outputActivation_ = self.outputActivationObj_.f

      self.recurrent_ = recurrent

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
      self.valueIn_ = None
      self.weight_ = self.iweight_( ( self.valueInSize_, self.valueOutSize_ ), self.nParameters_)
      self.valueOut_ = None
   
   def partial_fit( self, X, Y, maxBatch=None, elide=[] ):
      '''Perform single round of training without reset. Returns average error.
         max_batch indicates the number of sample from X to run at once. Exactly one update step is taken.'''
      grad, loss = np.zeros_like( self.weight_.x_ ), 0
      for bx, by in batch( X, Y, maxBatch ):
         portion = float( len( bx ) ) / len( X )
         bgrad, bloss = self.trainGrad_( bx, by, elide )
         grad += bgrad * portion
         loss = bloss * portion
      grad += self.regularizeGrad_()
      self.weight_ = self.update_( self.weight_, grad )
      return np.average( loss )

   def predict( self, X, maxBatch=None, elide=[] ):
      '''Perform forward pass.
      idempotent when recurrent=False.
      use elide list to leave out nodes (zero out, no update)'''
      if maxBatch is None:
         maxBatch = len( X )
      if X.shape[ 1 ] == self.inputSize_ - 1:
         X = np.append( np.ones( ( X.shape[ 0 ], 1 ) ), X, 1 )
      if not isinstance( elide, list ):
         elide = [ elide ]

      yPred = Dual( np.zeros( ( len( X ), self.outputSize_ ) ), 0, self.weight_.n_ )
      lower = 0
      by = None
      for bx in batch( X ):
         if self.recurrent_:
            by = self.predictRecurrent_( bx, elide )
         else:
            by = self.predictIndependent_( bx, elide )
         upper = lower + len( bx )
         yPred[ lower:upper ] = by
         lower += maxBatch
      return yPred
   
   def predictIndependent_( self, X, elide ):
      self.valueOut_ = Dual( np.zeros( ( len( X ), self.valueOutSize_ ) ), 0, self.nParameters_ )
      if X.shape[ 1 ] < self.valueInSize_:
         X = np.append( X, np.zeros( ( X.shape[ 0 ], self.valueInSize_ - X.shape[ 1 ] ) ), 1 )
      self.valueIn_ = Dual( X, 0, self.nParameters_ )
      for n in range( self.valueOutSize_ ):
         if n in elide:
            continue
         m = n + 1
         self.valueOut_[ :, n:m] = self.valueIn_.matmul( self.weight_[ :, n:m ] )
         if n < self.hiddenSize_:
            self.valueIn_[ :, self.inputSize_ + n:self.inputSize_ + m ] = self.hiddenActivation_( self.valueOut_[ :, n:m ] )
      return self.outputActivation_( self.valueOut_[ :, self.hiddenSize_: ] )

   def predictRecurrent_( self, X, elide ):
      pred = np.zeros( ( len( X ), self.outputSize_ ) )
      if not self.valueIn_:
         self.valueIn_ = Dual( np.zeros( self.valueInSize_ ), 0, self.nParameters_ )
         self.valueIn_[ 0 ] = 1
         self.valueOut_ = Dual( np.zeros( self.valueOutSize_ ), 0, self.nParameters_ )
      for s in range( len( X ) ):
         self.valueIn_[ :self.inputSize_ ] = X[ s ]
         for n in range( self.valueOutSize_ ):
            if n in elide:
               continue
            m = n + 1
            self.valueOut_[ n:m ] = self.valueIn_.matmul( self.weight_[ :, n:m ] )
            self.valueIn_[ self.inputSize_ + n:self.inputSize_ + m ] = self.hiddenActivation_( self.valueOut_[ n:m ] )
         pred[ s ] = self.outputActivation_( self.valueOut_[ self.hiddenSize_: ] )
      return pred

   def error( self, X, Y, maxBatch=None, elide=[] ):
      '''X and y should both be 2d numpy arrays'''
      pred = self.predict( X, elide=elide )
      return self.errorFunc_( Y, pred )

   def regularizeGrad_( self ):
      '''return gradient for regularization of weights in same shape as weights'''
      grad = self.regLoss_( self.weight_ ).e_
      while len( grad.shape ) > 1:
         grad = np.sum( grad, 1 )
      return grad.reshape( self.weight_.x_.shape )
   
   def trainGrad_( self, X, Y, elide ):
      '''return gradient for training weights in same shape as weights as well as the average loss'''
      loss = self.error( X, Y, elide=elide )
      grad = loss.e_
      grad = np.average( grad, 1 ) # across samples
      while len( grad.shape ) > 1:
         grad = np.sum( grad, 1 ) # across outputs
      return grad.reshape( self.weight_.x_.shape ), np.average( loss.x_ )

   # def addNode( self ):
   #    self.valueIn_ = np.insert( self.valueIn_, self.inputSize_, 0 )
   #    self.oldValueIn_ = self.valueIn_.copy()
   #    self.valueOut_ = np.insert( self.valueOut_, 0, 0 )
   #    self.dValueOut_ = np.insert( self.dValueOut_, 0, 0 )
   #    self.weight_ = np.insert( self.weight_, 0, np.random.normal( scale=np.sqrt( self.valueInSize_ ),
   #                                                                 size=self.valueInSize_ ), axis=0 )
   #    self.weight_ = np.insert( self.weight_, self.inputSize_, 0, axis=1 ) # np.random.normal( size=self.valueOutSize_ + 1 ), axis=1 )
   #    self.dWeight_ = np.insert( self.dWeight_, 0, 0, axis=0 )
   #    self.dWeight_ = np.insert( self.dWeight_, self.inputSize_, 0, axis=1 )
   #    self.hiddenSize_ += 1

   #    self.valueInSize_ = self.inputSize_ + self.hiddenSize_
   #    self.valueOutSize_ = self.hiddenSize_ + self.outputSize_

   #    self.updateObj_.add( 0, self.inputSize_ )
   
   # def removeNode( self, index ):
   #    self.valueIn_ = np.delete( self.valueIn_, self.inputSize_ + index )
   #    self.oldValueIn_ = self.valueIn_.copy()
   #    self.valueOut_ = np.delete( self.valueOut_, index )
   #    self.dValueOut_ = np.delete( self.dValueOut_, index )
   #    self.weight_ = np.delete( self.weight_, index, axis=0 )
   #    self.weight_ = np.delete( self.weight_, self.inputSize_ + index, axis=1 )
   #    self.dWeight_ = np.delete( self.dWeight_, index, axis=0 )
   #    self.dWeight_ = np.delete( self.dWeight_, self.inputSize_ + index, axis=1 )
   #    self.hiddenSize_ -= 1

   #    self.valueInSize_ = self.inputSize_ + self.hiddenSize_
   #    self.valueOutSize_ = self.hiddenSize_ + self.outputSize_

   #    self.updateObj_.remove( index, self.inputSize_ + index )

   def cool( self ):
      self.updateObj_.cool()

# def seqtrain( est, X, y, batch, trials=1 ):
#    error = np.zeros( est.maxIter_ )
#    for _ in range( trials ):
#       est.reset()
#       error += np.array( est.fit( X, y, batch ) )
#    return error / trials


# def growtrain( est, X, y, batch, rounds, maxLossInc=1.001 ):
#    '''grow and prune estimator during training'''
#    error = []
#    # replace with early halting
#    for r in range( rounds ):
#       if r:
#          #prune( est, X, y, maxLossInc )
#          est.addNode()
#       for _ in range( est.maxIter_ ):
#          error.append( np.average( est.partial_fit( X, y, batch ) ) )
#       print( 'Season %d/%d Complete, Loss %4g, Size %d' % ( r + 1, rounds, error[ -1 ], est.hiddenSize_ ) )
#    return error