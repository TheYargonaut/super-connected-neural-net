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

class L2( object ):
   def __init__( self, strength=1e-4 ):
      self.strength_ = strength
      self.dStrength_ = strength * 2

   def f( self, weight ):
      return self.strength_ * np.power( weight, 2 )

   def df( self, weight ):
      return self.dStrength_ * weight