from DualNumber.DualArithmetic import DualNumber as Dual, DualNumpy

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
      if isinstance( diff, Dual ):
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
      if isinstance( diff, Dual ):
         return full.where( condition, limited )
      return np.where( condition, limited, full )

   def df( self, target, predicted ):
      full = predicted - target
      high = full > self.delta_
      low = full < -self.delta_
      if isinstance( full, Dual ):
         return full.where( high, self.delta_ ).where( low, -self.delta_ )
      return np.where( low, -self.delta_, np.where( high, self.delta_, full ) )

class Hinge( Error ):
   def __init__( self ):
      pass

   def f( self, target, predicted ):
      t = np.sign( target )
      loss = 1 - predicted * t
      if isinstance( loss, Dual ):
         return loss.where( loss < 0, np.zeros_like( loss.x_ )  )
      return np.where( loss > 0, loss, np.zeros_like( loss ) )

   def df( self, target, predicted ):
      t = np.sign( target )
      loss = 1 - predicted * t
      if isinstance( loss, Dual ):
         loss = loss.x_
      return np.where( loss > 0, np.ones_like( loss ), np.zeros_like( loss ) )

class CrossEntropy( Error ):
   def __init__( self ):
      pass

   def f( self, target, predicted ):
      if isinstance( predicted, Dual ):
         return - ( target * predicted.log() )
      return -( target * np.log( predicted ) )

   def df( self, target, predicted ):
      return -( target / predicted )

class KlDivergence( Error ):
   '''Kullback-Liebler'''
   def __init__( self ):
      pass

   def f( self, target, predicted ):
      if isinstance( predicted, Dual ):
         return target * ( target / predicted ).log()
      return target * np.log( target / predicted )

   def df( self, target, predicted ):
      return -( target / predicted )

class JsDivergence( Error ):
   '''Jensen-Shannon'''
   def __init__( self ):
      pass

   def f( self, target, predicted ):
      m = ( target + predicted ) / 2
      if isinstance( m, Dual ):
         return ( target * ( target / m ).log() + predicted * ( predicted / m ).log() ) / 2
      return ( target * np.log( target / m ) + predicted * np.log( predicted / m ) ) / 2

   def df( self, target, predicted ):
      if not isinstance( predicted, Dual ):
         predicted = DualNumpy( predicted, 1 )
      return self.f( target, predicted ).e_


# TODO: Classification loss functions: Multinomial hinge (max, sum)