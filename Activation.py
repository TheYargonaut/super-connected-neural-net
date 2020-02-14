from DualNumber.DualArithmetic import DualNumber as Dual 
import numpy as np

class Activation( object ):
   def __init__( self ):
      '''abstract class'''
      raise NotImplementedError()

   def f( self, value ):
      raise NotImplementedError()

   def df( self, value ):
      raise NotImplementedError()

class Identity( object ):
   def __init__( self ):
      pass

   def f( self, value ):
      return value

   def df( self, value ):
      if isinstance( value, Dual ):
         return np.full_like( value.x_, 1 )
      return np.full_like( value, 1 )

class Relu( object ):
   def __init__( self ):
      pass

   def f( self, value ):
      if isinstance( value, Dual ):
         return value.where( value < 0, 0 )
      return np.where( value < 0, 0, value )

   def df( self, value ):
      if isinstance( value, Dual ):
         return np.where( value < 0, 0, np.full_like( value.x_, 1 ) )
      return np.where( value < 0, 0, np.full_like( value, 1 ) )

class Logistic( object ):
   def __init__( self ):
      pass

   def f( self, value ):
      if isinstance( value, Dual ):
         return 1 / ( 1 + ( -value ).exp() )
      return 1 / ( 1 + np.exp( -value ) )

   def df( self, value ):
      lf = self.f( value )
      return lf * ( 1 - lf )

class Tanh( object ):
   def __init__( self ):
      pass

   def f( self, value ):
      if isinstance( value, Dual ):
         return value.tanh()
      return np.tanh( value )

   def df( self, value ):
      return 1 - self.f( value ) ** 2

# TODO: Softmax, Exponential Linear Unit (ELU), leaky RELU