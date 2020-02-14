from DualNumber.DualArithmetic import DualNumber

import numpy as np

class Error( object ):
   def __init__( self ):
      '''abstract class'''
      raise NotImplementedError()

   def f( self, target, predicted ):
      raise NotImplementedError()

   def df( self, target, predicted ):
      raise NotImplementedError()

class Mae( Error ):
   '''Mean absolute error'''
   def __init__( self ):
      pass

   def f( self, target, predicted ):
      return abs( predicted - target )

   def df( self, target, predicted ):
      diff = predicted - target
      if isinstance( predicted, diff ):
         return np.sign( diff.x_ )
      return np.sign( predicted - target )

class Mse( Error ):
   def __init__( self ):
      pass

   def f( self, target, predicted ):
      return ( predicted - target ) ** 2

   def df( self, target, predicted ):
      return ( predicted - target ) * 2

class Huber( Error ):
   def __init__( self, delta=1 ):
      '''abstract class'''
      self.delta_ = delta

   def f( self, target, predicted ):
      diff = abs( predicted - target )
      full = ( diff ** 2 ) * 0.5
      limited = ( diff - 0.5 * self.delta_ ) * self.delta_
      condition = diff > self.delta_
      if isinstance( diff, DualNumber ):
         return full.where( condition, limited )
      return np.where( condition, limited, full )

   def df( self, target, predicted ):
      full = predicted - target
      high = full > self.delta_
      low = full < -self.delta_
      if isinstance( full, DualNumber ):
         return full.where( high, self.delta_ ).where( low, -self.delta_ )
      return np.where( low, -self.delta_, np.where( high, self.delta_, full ) )

# TODO: Classification loss functions: cross-entropy, Kullback-Liebler, hinge