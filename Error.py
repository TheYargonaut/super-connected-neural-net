from DualNumber.DualArithmetic import DualNumber as Dual, DualNumpy

import numpy as np

# to avoid zeros in logarithms
inft = 1e-18

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
         return loss.where( loss < 0, 0 )
      return np.where( loss > 0, loss, 0 )

   def df( self, target, predicted ):
      t = np.sign( target )
      loss = 1 - predicted * t
      if isinstance( loss, Dual ):
         loss = loss.x_
      return np.where( loss > 0, 1, 0 )

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
         return target * ( target / predicted ).log().where( target <= 0, 0 )
      return target * np.where( target > 0, np.log( target / predicted + inft ), 0 )

   def df( self, target, predicted ):
      return -( target / predicted )

class JsDivergence( Error ):
   '''Jensen-Shannon'''
   def __init__( self ):
      pass

   def f( self, target, predicted ):
      m = ( predicted + target ) / 2
      if isinstance( m, Dual ):
         target = type(m)( target, 0, m.n_ )
         tpart = target * ( target / m ).log().where( target <= 0, 0 )
         ppart = predicted * ( predicted / m ).log().where( predicted <= 0, 0 )
         return ( tpart + ppart ) / 2
      tpart = target * np.where( target > 0, np.log( target / m ), 0 )
      ppart = predicted * np.where( predicted > 0, np.log( predicted / m ), 0 )
      return ( tpart + ppart ) / 2

   def df( self, target, predicted ):
      if not isinstance( predicted, Dual ):
         predicted = DualNumpy( predicted, 1 )
      return self.f( target, predicted ).e_


# TODO: Classification loss functions: Multinomial hinge (max, sum)