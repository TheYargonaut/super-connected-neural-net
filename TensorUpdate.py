from DualNumber.DualArithmetic import DualNumber, DualGrad

import tensorflow as tf
import pdb

class Update( object ):
   def __init__( self ):
      '''abstract class'''
      raise NotImplementedError()

   def step( self, weight, gradient ):
      '''returns the result of updating the weight based on the gradient'''
      raise NotImplementedError()
   
   def reset( self ):
      '''purge internal state'''
      pass
   
   def add( self, row, column ):
      '''add space to internal state'''
      pass
   
   def remove( self, row, column ):
      '''remove space from internal state'''
      pass

   def cool( self ):
      '''attenuate learning-rate or learning-rate equivalent'''
      pass

class Sgd( Update ):
   '''stochastic gradient descent. no internal state beyond learning rate'''
   def __init__( self, learningRate=0.001, decay=1 ):
      self.learningRate_ = -learningRate
      self.decay_ = decay

   def step( self, weight, gradient ):
      delta = tf.math.multiply( gradient, self.learningRate_ )
      return tf.math.add( weight, delta )

   def cool( self ):
      self.learningRate_ *= self.decay_

class Rprop( Update ):
   '''resillient backpropagation'''
   def __init__( self, increase=1.2, decrease=0.5 ):
      self.increase_ = increase
      self.decrease_ = decrease
      self.reset()

   def step( self, weight, gradient ):
      if self.speed_ is None:
         self.speed_ = tf.ones_like( gradient )
         self.signs_ = tf.zeros_like( gradient )
      tempSign = tf.math.sign( gradient )
      # decrease where different, increase where similar
      accelerate = tf.math.logical_and( tempSign == self.signs_, abs( gradient ) > 1e-16 )
      self.speed_ *= tf.where( accelerate, self.increase_, self.decrease_ )
      self.signs_ = tempSign
      return tf.subtract( weight, tf.multiply( self.speed_, self.signs_ ) )
   
   def reset( self ):
      self.speed_ = None
      self.signs_ = None
