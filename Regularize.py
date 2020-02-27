from DualNumber.DualArithmetic import DualNumber as Dual
import numpy as np

class Regularize( object ):
   def __init__( self ):
      '''abstract class'''
      raise NotImplementedError()

   def f( self, weight ):
      raise NotImplementedError()

   def df( self, weight ):
      raise NotImplementedError()
   
class Zero( object ):
   def __init__( self ):
      pass

   def f( self, weight ):
      return 0

   def df( self, weight ):
      return 0

class Ridge( object ):
   def __init__( self, strength=1e-4 ):
      self.strength_ = strength
      self.dStrength_ = strength * 2

   def f( self, weight ):
      if isinstance( weight, Dual ):
         return ( weight ** 2 ) * self.strength_ / weight.x_.size
      return self.strength_ * np.power( weight, 2 ) / weight.size

   def df( self, weight ):
      return weight * self.dStrength_ / getattr( weight, 'x_', weight ).size

# TODO
#class Lasso
#class ElasticNet