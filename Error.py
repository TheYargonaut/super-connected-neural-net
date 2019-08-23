import numpy as np

class Error( object ):
   def __init__( self ):
      '''abstract class'''
      raise NotImplementedError()

   def f( self, target, predicted ):
      raise NotImplementedError()

   def df( self, target, predicted ):
      raise NotImplementedError()

class Mse( Error ):
   def __init__( self ):
      pass

   def f( self, target, predicted ):
      return ( predicted - target )**2

   def df( self, target, predicted ):
      return 2 * ( predicted - target )

class Huber( Error ):
   def __init__( self, limit=1 ):
      '''abstract class'''
      self.limit_ = limit
      self.dlimit_ = 2 * np.sqrt( limit )

   def f( self, target, predicted ):
      return np.minimum( np.maximum( -self.limit_, ( predicted - target )**2 ), self.limit_ )

   def df( self, target, predicted ):
      return np.minimum( np.maximum( -self.dlimit_, 2 * ( predicted - target ) ), self.dlimit_ )