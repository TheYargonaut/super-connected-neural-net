# import numpy as np
# import Error, Update, Activation, Regularize
# from Network import SNN_Regressor

# helper training functions for snn

def batch( X, Y=None, maxBatch=None ):
   '''Produce slices of X of size at most maxBatch for iterating'''
   if not maxBatch:
      maxBatch = len( X )
   if Y is None:
      for i in range( 0, len( X ), maxBatch ):
         upper = min( i + maxBatch, len( X ) )
         yield X[ i : upper ]
   else:
      for i in range( 0, len( X ), maxBatch ):
         upper = min( i + maxBatch, len( X ) )
         yield X[ i : upper ], Y[ i : upper ]

# def prune( est, X, y, maxLossInc=1.001, loss=Error.Mse() ):
#    lossPrior = np.average( loss.f( y, est.predict( X ) ) )
#    maxLoss = lossPrior * maxLossInc
#    while est.hiddenSize_ > 0:
#       best = None
#       for hn in range( est.hiddenSize_ ):
#          lossElide = np.average( loss.f( y, est.predict( X, hn ) ) )
#          if lossElide < maxLoss and ( best is None or lossElide < best[ 0 ] ):
#             best = ( lossElide, hn )
#       if best is None:
#          return
#       else:
#          est.removeNode( best[ 1 ] )

# def grow( est, n ):
#    for _ in range( n ):
#       est.addNode()

# list of ordered tuples (wait, function on est)
# def train( est, X, y, batch=0, verbose=True, maxIter=1000, stopWait=0, triggers=[] ):
#    '''returns trace of error. Use pairs in triggers to alter where hung up'''
#    lossLog = [ 0 ] * ( maxIter + 1 )
#    bestLoss = np.average( est.error( X, y ) )
#    lossLog[ 0 ] = bestLoss
   
#    improvementWait = 0
#    for step in range( maxIter ):
#       loss = np.average( est.partial_fit( X, y, batch=batch ) )
#       lossLog[ step + 1 ] = loss
      
#       if loss < 0.99 * bestLoss:
#          bestLoss = loss
#          improvementWait = 0
#       else:
#          improvementWait += 1
      
#          if stopWait and improvementWait >= stopWait:
#             return lossLog[ : step + 1 ]
         
#          for t in triggers:
#             if improvementWait == t[0]:
#                t[1]( est )
      
#       if verbose:
#          print( 'Step %d/%d Complete, Loss %4g, Size %d' % ( step + 1, maxIter, loss, est.hiddenSize_ ) )
#    return lossLog