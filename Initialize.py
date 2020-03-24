from DualNumber.DualArithmetic import DualGrad as Dual
import numpy as np

def makeInft( shape, total, first ):
   '''shape should be iterable'''
   inft = np.identity( total )
   inft = inft[ :, first : first + np.product( shape ) ]
   return np.reshape( inft, ( total, *shape ) )

def zero( shape, total, first=0 ):
   real = np.zeros( shape )
   inft = makeInft( real.shape, total, first )
   return Dual( real, inft, total )

def normal( shape, total, first=0, mean=0, std=1 ):
   '''randomize initial weights'''
   real = np.random.standard_normal( shape ) * std + mean
   inft = makeInft( real.shape, total, first )
   return Dual( real, inft, total )

def he( shape, total, first=0, n=None ):
   '''for relu.
   n should be cardinality of parameters in layer or of sum of inputs and outputs for layer'''
   if not n:
      n = total
   std = np.sqrt( 2 / n )
   return normal( shape, total, first, std=std )

def xavier( shape, total, first=0, n=None ):
   '''for tanh.
   n should be cardinality of parameters in layer or of sum of inputs and outputs for layer'''
   if not n:
      n = total
   std = np.sqrt( 1 / n )
   return normal( shape, total, first, std=std )

def lecun_normal( shape, total, first=0, n=None ):
   '''for selu.
   n should be cardinality of inputs for layer'''
   if not n:
       n = total
   std = np.sqrt( 1 / n ) / 0.87962566103423978
   return normal( shape, total, first, std=std )
