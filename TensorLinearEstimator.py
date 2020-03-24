# Linear estimator using dual number and Tensorflow

import TensorUpdate, TensorLoss, TensorRegularizer, TensorActivation, TensorInitializer
from DualNumber import TensorDual as td
from DualNumber.DualArithmetic import DualGrad as Dual
from Train import batch

import copy
import tensorflow as tf
import pdb

class TLE( object ):
   '''Linear Least-Squares estimator using gradient descent with dual numbers'''
   _estimator_type = "regressor"

   def __init__( self, inputSize, outputSize,
                 outputAct=TensorActivation.identity,
                 iweight=TensorInitializer.zero,
                 update=TensorUpdate.Sgd( 1e-6 ),
                 error=TensorLoss.mse,
                 regularization=TensorRegularizer.zero ):
      # structure settings
      self.inputSize_ = inputSize
      self.outputSize_ = outputSize

      self.nParameters_ = tf.constant( self.inputSize_ * self.outputSize_ + 1 )

      self.outputActivation_ = outputAct

      # training settings
      self.iweight_ = iweight
      self.errorFunc_ = error
      self.wUpdateObj_ = copy.deepcopy( update )
      self.bUpdateObj_ = copy.deepcopy( update )
      self.wUpdate_ = self.wUpdateObj_.step
      self.bUpdate_ = self.bUpdateObj_.step
      self.regLoss_ = regularization

      # put in clean state
      self.initialize_()

   def reset( self ):
      self.wUpdateObj_.reset()
      self.bUpdateObj_.reset()
      self.initialize_()
   
   def initialize_( self ):
      self.weight_, self.wInft_ = self.iweight_( ( self.inputSize_, self.outputSize_ ), self.nParameters_, 1 )
      self.bias_, self.bInft_ = self.iweight_( ( 1, 1 ), self.nParameters_, 0 )
      # TODO: regularize bias
   
   def partial_fit( self, xReal, xInft, yReal, yInft ):
      '''Perform single round of training without reset. Returns average loss and gradient.'''
      pred = self.trainGrad_( xReal, xInft, yReal, yInft )
      reg = self.regularizeGrad_()
      loss = td.add( *pred, *reg )
      self.weight_ = self.wUpdate_( self.weight_, tf.reshape( loss[ 1 ][ 1: ], self.weight_.shape ) )
      self.bias_ = self.bUpdate_( self.bias_, tf.reshape( loss[ 1 ][ 0:1 ], self.bias_.shape ) )
      return loss

   def predict( self, xReal, xInft ):
      '''X should be rank 2 tensor'''
      yReal, yInft = td.add( *td.tensordot( xReal, xInft, self.weight_, self.wInft_ ), self.bias_, self.bInft_ )
      yInft = tf.transpose( yInft, [ 0, 2, 1 ] )
      return self.outputActivation_( yReal, yInft )
   
   def loss( self, xReal, xInft, yReal, yInft ):
      '''X and y should both be rank 2 tensors'''
      p = self.predict( xReal, xInft )
      return self.errorFunc_( yReal, yInft, *p )

   def regularizeGrad_( self ):
      '''return gradient for regularization of weights in same shape as weights along with average loss'''
      loss = self.regLoss_( self.weight_, self.wInft_ )
      bloss = self.regLoss_( self.bias_, self.bInft_ )
      real = ( tf.math.reduce_sum( loss[ 0 ] ) + bloss[ 0 ] ) / tf.cast( self.nParameters_, dtype=tf.float32 )
      inft = loss[ 1 ] # bloss?
      if len( inft.shape ) > 1:
         elimaxes = [ a for a in range( len( loss[ 1 ].shape ) ) if a > 0 ]
         inft = tf.math.reduce_mean( inft, elimaxes )
      return real, inft
   
   def trainGrad_( self, xReal, xInft, yReal, yInft ):
      '''return gradient for training weights in same shape as weights as well as the average loss'''
      loss = self.loss( xReal, xInft, yReal, yInft )
      real = tf.math.reduce_mean( loss[ 0 ] )
      elimaxes = [ a for a in range( len( loss[ 1 ].shape ) ) if a > 0 ]
      inft = tf.math.reduce_mean( loss[ 1 ], elimaxes )
      return real, inft

   def cool( self ):
      self.wUpdateObj_.cool()
      self.bUpdateObj_.cool()