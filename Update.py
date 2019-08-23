import numpy as np
import pdb

def addRowCol( arr, row, column, value=0 ):
   '''delete row and column of array provided'''
   return np.insert( np.insert( arr, row, value, axis=0 ), column, value, axis=1 )
def delRowCol( arr, row, column ):
   '''delete row and column of array provided'''
   return np.delete( np.delete( arr, row, axis=0 ), column, axis=1 )

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
      delta = gradient * self.learningRate_
      return weight + delta

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
         self.speed_ = np.ones_like( weight )
         self.signs_ = np.zeros_like( weight )
      tempSign = np.sign( gradient )
      # decrease where different, increase where similar
      self.speed_ *= np.where( np.logical_and( np.equal( tempSign, self.signs_ ), 
                                               np.not_equal( tempSign, np.zeros_like( tempSign ) ) ), 
                               self.increase_, self.decrease_ )
      self.signs_ = tempSign
      return weight - self.speed_ * self.signs_
   
   def reset( self ):
      self.speed_ = None
      self.signs_ = None
   
   def add( self, row, column ):
      if self.speed_ is None:
         return
      self.speed_ = addRowCol( self.speed_, row, column, 1 )
      self.signs_ = addRowCol( self.signs_, row, column, 1 )
   
   def remove( self, row, column ):
      if self.speed_ is None:
         return
      self.speed_ = delRowCol( self.speed_, row, column )
      self.signs_ = delRowCol( self.signs_, row, column )

class Momentum( Update ):
   '''stochastic gradient descent'''
   def __init__( self, learningRate=0.001, momentum=0.9, rateDecay=1, frictionDecay=1 ):
      self.learningRate_ = -learningRate
      self.momentum_ = momentum
      self.rateDecay_ = rateDecay
      self.frictionDecay_ = frictionDecay
      self.reset()

   def step( self, weight, gradient ):
      if self.velocity_ is None:
         self.velocity_ = np.zeros_like( weight )
      self.velocity_ = ( self.velocity_ * self.momentum_ ) + ( gradient * self.learningRate_ )
      return weight + self.velocity_
   
   def reset( self ):
      self.velocity_ = None
   
   def add( self, row, column ):
      if self.velocity_ is None:
         return
      self.velocity_ = addRowCol( self.velocity_, row, column )
   
   def remove( self, row, column ):
      if self.velocity_ is None:
         return
      self.velocity_ = delRowCol( self.velocity_, row, column )
   
   def cool( self ):
      self.learningRate_ *= self.rateDecay_
      self.momentum_ = 1 - ( 1 - self.momentum_ ) * self.frictionDecay_

class NesterovMomentum( Update ):
   '''stochastic gradient descent'''
   def __init__( self, learningRate=0.001, momentum=0.9, rateDecay=1, frictionDecay=1 ):
      self.learningRate_ = -learningRate
      self.momentum_ = momentum
      self.rateDecay_ = rateDecay
      self.frictionDecay_ = frictionDecay
      self.reset()

   def step( self, weight, gradient ):
      if self.velocity_ is None:
         self.velocity_ = np.zeros_like( weight )
      self.oldVelocity_ = self.velocity_
      self.velocity_ = ( self.velocity_ * self.momentum_ ) + ( gradient * self.learningRate_ )
      return weight - self.oldVelocity_ + ( 1 + self.momentum_ ) * self.velocity_
   
   def reset( self ):
      self.velocity_ = None
   
   def add( self, row, column ):
      if self.velocity_ is None:
         return
      self.velocity_ = addRowCol( self.velocity_, row, column )
   
   def remove( self, row, column ):
      if self.velocity_ is None:
         return
      self.velocity_ = delRowCol( self.velocity_, row, column )

   def cool( self ):
      self.learningRate_ *= self.rateDecay_
      self.momentum_ = 1 - ( 1 - self.momentum_ ) * self.frictionDecay_

class RmsProp( Update ):
   '''stochastic gradient descent'''
   def __init__( self, learningRate=0.001, rho=0.9, momentum=0.0, epsilon=1e-8, centered=False, rateDecay=1, momentDecay=1 ):
      self.learningRate_ = -learningRate
      self.rho_ = rho
      self.momentum_ = momentum
      self.epsilon_ = epsilon
      self.centered_ = centered
      self.rateDecay_ = rateDecay
      self.momentDecay_ = momentDecay
      self.reset()

   def step( self, weight, gradient ):
      if self.meanSquare_ is None:
         self.meanSquare_ = np.zeros_like( weight )
         self.mean_ = np.zeros_like( weight )
         self.velocity_ = np.zeros_like( weight )
      self.meanSquare_ = self.rho_ * self.meanSquare_ + ( 1 - self.rho_ ) * np.power( gradient, 2 )
      if self.centered_:
         self.mean_ = self.rho_ * self.mean_ + (1 - self.rho_) * gradient
         self.velocity_ = self.momentum_ * self.velocity_ + self.learningRate_ * gradient / np.sqrt( self.meanSquare_ - np.power( self.mean_, 2 ) + self.epsilon_ )
      else:
         self.velocity_ = self.momentum_ * self.velocity_ + self.learningRate_ * gradient / np.sqrt( self.meanSquare_ + self.epsilon_ )
      return weight + self.velocity_
   
   def reset( self ):
      self.meanSquare_ = None
      self.mean_ = None
      self.velocity_ = None
   
   def add( self, row, column ):
      if self.meanSquare_ is None:
         return
      self.meanSquare_ = addRowCol( self.meanSquare_, row, column )
      self.mean_ = addRowCol( self.mean_, row, column )
      self.velocity_ = addRowCol( self.velocity_, row, column )
   
   def remove( self, row, column ):
      if self.meanSquare_ is None:
         return
      self.meanSquare_ = delRowCol( self.meanSquare_, row, column )
      self.mean_ = delRowCol( self.mean_, row, column )
      self.velocity_ = delRowCol( self.velocity_, row, column )
   
   def cool( self ):
      self.learningRate_ *= self.rateDecay_
      self.momentDecay_ *= self.momentDecay_

class Adam( Update ):
   '''stochastic gradient descent'''
   def __init__( self, learningRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=1 ):
      self.learningRate_ = -learningRate
      self.beta1_ = beta1
      self.beta1c_ = 1 - beta1
      self.beta1pow_ = beta1
      self.beta2_ = beta2
      self.beta2c_ = 1 - beta2
      self.beta2pow_ = beta2
      self.epsilon_ = epsilon
      self.decay_ = decay
      self.reset()

   def step( self, weight, gradient ):
      if self.m_ is None:
         self.m_ = np.zeros_like( weight )
         self.v_ = np.zeros_like( weight )
      self.m_ = self.beta1_ * self.m_ + self.beta1c_ * gradient
      self.v_ = self.beta2_ * self.v_ + self.beta2c_ * gradient * gradient
      mHat = self.m_ / ( 1 - self.beta1pow_ )
      vHat = self.v_ / ( 1 - self.beta2pow_ )
      self.beta1pow_ *= self.beta1_
      self.beta2pow_ *= self.beta2_
      delta = self.learningRate_ * mHat / ( np.sqrt( vHat ) + self.epsilon_ )
      return weight + delta
   
   def reset( self ):
      self.m_ = None
      self.v_ = None
   
   def add( self, row, column ):
      if self.m_ is None:
         return
      self.m_ = addRowCol( self.m_, row, column )
      self.v_ = addRowCol( self.v_, row, column )
   
   def remove( self, row, column ):
      if self.m_ is None:
         return
      self.m_ = delRowCol( self.m_, row, column )
      self.v_ = delRowCol( self.v_, row, column )
   
   def cool( self ):
      self.learningRate_ *= self.decay_