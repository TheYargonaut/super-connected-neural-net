from DualNumber.DualArithmetic import DualNumber as Dual, DualNumpy
import numpy as np
import pdb

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
         return np.ones_like( value.x_ )
      return np.ones_like( value )

class Softplus( object ):
   '''smooth approximation of rectifier'''

   def __init__( self ):
      pass

   def f( self, value ):
      if isinstance( value, Dual ):
         return ( 1 + value.exp() ).log()
      return np.log( 1 + np.exp( value ) )
   
   def df( self, value ):
      if isinstance( value, Dual ):
         return 1 / ( 1 + ( -value ).exp() )
      return 1 / ( 1 + np.exp( -value ) )

class Relu( object ):
   def __init__( self ):
      pass

   def f( self, value ):
      if isinstance( value, Dual ):
         return value.where( value < 0, 0 )
      return np.where( value < 0, 0, value )

   def df( self, value ):
      if isinstance( value, Dual ):
         return np.where( value < 0, 0, 1 )
      return np.where( value < 0, 0, 1 )

class LeakyRelu( object ):
   def __init__( self, p=0.01 ):
      self.p_ = p

   def f( self, value ):
      low = value * self.p_
      if isinstance( value, Dual ):
         return value.where( value < 0, low )
      return np.where( value < 0, low, value )

   def df( self, value ):
      if isinstance( value, Dual ):
         return np.where( value < 0, self.p_, 1 )
      return np.where( value < 0, self.p_, 1 )

class Elu( object ):
   def __init__( self ):
      pass

   def f( self, value ):
      if isinstance( value, Dual ):
         low = value.exp() - 1
         return value.where( value < 0, low )
      low = np.exp( value ) - 1
      return np.where( value < 0, low, value )
   
   def df( self, value ):
      if isinstance( value, Dual ):
         return np.where( value < 0, np.exp( value.x_ ) , 1 )
      return np.where( value < 0, np.exp( value ), 1 )

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

class Softmax( object ):
   def __init__( self, temperature=1 ):
      self.t_ = temperature

   def f( self, value ):
      if isinstance( value, Dual ):
         raw = ( value / self.t_ ).exp()
         return raw / raw.sum( -1 ).reshape( ( *raw.x_.shape[:-1], 1 ) )
      raw = np.exp( value / self.t_ )
      return raw / np.sum( raw, -1 )

   def df( self, value ):
      if not isinstance( value, DualNumpy ):
         value = DualNumpy( value, 1 )
      return self.f( value ).e_
      