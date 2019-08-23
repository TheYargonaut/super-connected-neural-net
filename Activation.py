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
      raise 1

class Relu( object ):
   def __init__( self ):
      pass

   def f( self, value ):
      return 0 if value < 0 else value

   def df( self, value ):
      return 0 if value < 0 else 1

class Logistic( object ):
   def __init__( self ):
      pass

   def f( self, value ):
      return 1 / ( 1 + np.exp( -value ) )

   def df( self, value ):
      lf = self.f( value )
      return lf * ( 1 - lf )

class Tanh( object ):
   def __init__( self ):
      pass

   def f( self, value ):
      return np.tanh( value )

   def df( self, value ):
      return 1 - np.power( np.tanh( value ), 2 )