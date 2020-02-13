
import Update, Error, Regularize, Activation, Initialize
from DualNumber.DualArithmetic import DualGrad as Dual

import numpy as np
from sklearn.metrics import mean_squared_error
import pdb

# np.seterr(invalid='raise')

# version which uses Dual Numbers for automatic forward differentiation

# Separate trainer and model?

class SNN_Regressor( object ):
   _estimator_type = "regressor"

   def __init__( self, inputSize, outputSize, hiddenSize=100, maxIter=200, independent=True,
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

      self.independent_ = independent

      # training settings
      self.maxIter_ = maxIter
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
      self.valueIn_ = Dual( np.zeros( self.valueInSize_ ), 0, self.nParameters_ )
      self.valueIn_[ 0 ] = 1 # constant for input
      self.weight_ =  self.iweight_()
      self.dWeight_ = np.zeros_like( self.weight_.x_ )
      self.valueOut_ = Dual( np.zeros( self.valueOutSize_ ), 0, self.nParameters_ )
      self.output_ = Dual( np.zeros( self.outputSize_ ), 0, self.nParameters_ )
      
   def fit( self, X, y, batch=0, verbose=True ):
      '''Perform training for maxIter rounds. Returns sequence of error per round.'''
      self.reset()
      errorTrace = [ None ]*self.maxIter_
      for r in range( self.maxIter_ ):
         errorTrace[ r ] = np.average( self.partial_fit( X, y, batch ) )
         if verbose:
            print( 'Round %d/%d Complete, Loss %4g' % ( r + 1, self.maxIter_, errorTrace[ r ] ) )
      return errorTrace
   
   def partial_fit( self, X, y, batch=0 ):
      '''Perform single round of training without reset. Returns average error.'''
      if not batch:
         batch = X.shape[ 0 ]

      # ensure one sample per training round. Batch-average derivative
      if len( X.shape ) == 2:
         if len( y.shape ) == 1:
            y = y[ :, None ]
         errorSum = 0
         derivativeSum = np.zeros_like( self.weight_ )
         for n in range( X.shape[0] ):
            self.predict( X[ n, : ] )
            errorSum += self.errorFunc_( y[ n, : ], self.output_ )
            self.derivative_( y[ n, : ] )
            derivativeSum += self.dWeight_
            # batching
            if not ( ( n + 1 ) % batch ):
               self.weight_ = self.update_( self.weight_, ( derivativeSum / batch ) + 
                                                         self.dRegLoss_( self.weight_ ) )
               derivativeSum = np.zeros_like( self.weight_ )
         # deal with any odd ones out
         remainder = X.shape[ 0 ] % batch
         if remainder:
            self.weight_ = self.update_( self.weight_, ( derivativeSum / remainder ) + 
                                                      self.dRegLoss_( self.weight_ ) )
         return errorSum / X.shape[ 0 ]

      # only one sample given, do an update 
      self.predict( X )
      self.derivative_( y )
      self.dWeight_ += self.dRegLoss_( self.weight_ )
      self.weight_ = self.update_( self.weight_, self.dWeight_ )
      return self.errorFunc_( y, self.output_ )

   def predict( self, X, elide=[] ):
      '''Perform forward pass.
      idempotent only if independent specified True.
      use elide list to leave out nodes (zero out, no update)'''

      if not isinstance( elide, list ):
         elide = [ elide ]

      # ensure one sample per prediction
      if len( X.shape ) == 2:
         yPred = np.zeros( ( X.shape[0], self.outputSize_ ) )
         for n in range( X.shape[0] ):
            yPred[ n, : ] = self.predict_( X[ n, : ], elide )
         return yPred.reshape( -1 )
      return self.predict_( X, elide )
   
   def predict_( self, X, elide ):
      '''internal version, assumes and returns for single sample'''
      
      # enforce independence if specified
      if self.independent_:
         self.valueIn_ = np.zeros( self.valueInSize_ )
         self.valueIn_[ 0 ] = 1
      else:
         for h in elide:
            self.valueIn_[ h + self.inputSize_ ] = 0
      # input
      self.valueIn_[ 1:self.inputSize_ ] = X
      np.copyto( self.oldValueIn_, self.valueIn_ )
      # hidden nodes
      for n in range( self.hiddenSize_ ):
         if n not in elide:
            self.valueOut_[ n ] = np.dot( self.valueIn_, self.weight_[ n, : ] )
            self.valueIn_[ n + self.inputSize_ ] = self.hiddenActivation_( self.valueOut_[ n ] )
      # output nodes
      self.valueOut_[ self.hiddenSize_:self.valueOutSize_ ] = np.matmul(
         self.weight_[ self.hiddenSize_:self.valueOutSize_, : ], self.valueIn_ )
      self.output_ = self.outputActivation_( self.valueOut_[ self.hiddenSize_:self.valueOutSize_ ] )
      return self.output_

   def error( self, X, y, elide=[] ):
      if not isinstance( elide, list ):
         elide = [ elide ]
      # ensure one sample per error
      if len( X.shape ) == 2:
         yPred = np.zeros( ( X.shape[0], self.outputSize_ ) )
         for n in range( X.shape[0] ):
            yPred[ n, : ] = self.predict_( X[ n, : ], elide )
         return self.errorFunc_( yPred.reshape( -1 ), y.reshape( -1 ) )

   def addNode( self ):
      self.valueIn_ = np.insert( self.valueIn_, self.inputSize_, 0 )
      self.oldValueIn_ = self.valueIn_.copy()
      self.valueOut_ = np.insert( self.valueOut_, 0, 0 )
      self.dValueOut_ = np.insert( self.dValueOut_, 0, 0 )
      self.weight_ = np.insert( self.weight_, 0, np.random.normal( scale=np.sqrt( self.valueInSize_ ),
                                                                   size=self.valueInSize_ ), axis=0 )
      self.weight_ = np.insert( self.weight_, self.inputSize_, 0, axis=1 ) # np.random.normal( size=self.valueOutSize_ + 1 ), axis=1 )
      self.dWeight_ = np.insert( self.dWeight_, 0, 0, axis=0 )
      self.dWeight_ = np.insert( self.dWeight_, self.inputSize_, 0, axis=1 )
      self.hiddenSize_ += 1

      self.valueInSize_ = self.inputSize_ + self.hiddenSize_
      self.valueOutSize_ = self.hiddenSize_ + self.outputSize_

      self.updateObj_.add( 0, self.inputSize_ )
   
   def removeNode( self, index ):
      self.valueIn_ = np.delete( self.valueIn_, self.inputSize_ + index )
      self.oldValueIn_ = self.valueIn_.copy()
      self.valueOut_ = np.delete( self.valueOut_, index )
      self.dValueOut_ = np.delete( self.dValueOut_, index )
      self.weight_ = np.delete( self.weight_, index, axis=0 )
      self.weight_ = np.delete( self.weight_, self.inputSize_ + index, axis=1 )
      self.dWeight_ = np.delete( self.dWeight_, index, axis=0 )
      self.dWeight_ = np.delete( self.dWeight_, self.inputSize_ + index, axis=1 )
      self.hiddenSize_ -= 1

      self.valueInSize_ = self.inputSize_ + self.hiddenSize_
      self.valueOutSize_ = self.hiddenSize_ + self.outputSize_

      self.updateObj_.remove( index, self.inputSize_ + index )

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