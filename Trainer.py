import numpy as np
import Error, Update, Activation, Regularize
from Network import SNN_Regressor

# helper training functions for snn

def prune( est, X, y, maxLossInc=1.001, loss=Error.Mse() ):
   lossPrior = np.average( loss.f( y, est.predict( X ) ) )
   maxLoss = lossPrior * maxLossInc
   while est.hiddenSize_ > 0:
      best = None
      for hn in range( est.hiddenSize_ ):
         lossElide = np.average( loss.f( y, est.predict( X, hn ) ) )
         if lossElide < maxLoss and ( best is None or lossElide < best[ 0 ] ):
            best = ( lossElide, hn )
      if best is None:
         return
      else:
         est.removeNode( best[ 1 ] )

def grow( est, n ):
   for _ in range( n ):
      est.addNode()

# list of ordered tuples (wait, function on est)
def train( est, X, y, batch=0, verbose=True, maxIter=1000, stopWait=0, triggers=[] ):
   '''returns trace of error. Use pairs in triggers to alter where hung up'''
   lossLog = [ 0 ] * ( maxIter + 1 )
   bestLoss = np.average( est.error( X, y ) )
   lossLog[ 0 ] = bestLoss
   
   improvementWait = 0
   for step in range( maxIter ):
      loss = np.average( est.partial_fit( X, y, batch=batch ) )
      lossLog[ step + 1 ] = loss
      
      if loss < 0.99 * bestLoss:
         bestLoss = loss
         improvementWait = 0
      else:
         improvementWait += 1
      
         if stopWait and improvementWait >= stopWait:
            return lossLog[ : step + 1 ]
         
         for t in triggers:
            if improvementWait == t[0]:
               t[1]( est )
      
      if verbose:
         print( 'Step %d/%d Complete, Loss %4g, Size %d' % ( step + 1, maxIter, loss, est.hiddenSize_ ) )
   return lossLog

# unit test
if __name__ == "__main__":
   import matplotlib.pyplot as plt

   commonSettings = dict(
      inputSize=2,
      outputSize=1,
      hiddenSize=20,
      maxIter=1000,
      independent=True,
      error=Error.Mse(), 
      regularization=Regularize.Ridge(),
      hiddenAct=Activation.Tanh()
   )

   X = np.array( [ [ -1, 2 ], [ 2, -4 ], [ -3, 6 ] ] )
   y = np.array( [ [ 10 ], [ 40 ], [ 90 ] ] )

   commonTriggers = [
      ( 3, lambda e: e.cool() ), # cool
      ( 6, lambda e: prune( e, X, y ) ), # prune
      ( 9, lambda e: grow( e,  max( 1, 1 + int( np.log( e.hiddenSize_ + 1 ) ) ) ) ), # grow
   ]

   sgde = SNN_Regressor( **commonSettings, update=Update.Sgd( learningRate=1e-2, decay=0.9 ) )
   mome = SNN_Regressor( **commonSettings, update=Update.Momentum( learningRate=1e-2, rateDecay=0.9, frictionDecay=0.99 ) )
   nest = SNN_Regressor( **commonSettings, update=Update.NesterovMomentum( learningRate=1e-2, rateDecay=0.9, frictionDecay=0.99 ) )
   adam = SNN_Regressor( **commonSettings, update=Update.Adam( learningRate=1e-2, decay=0.9 ) )
   rmsp = SNN_Regressor( **commonSettings, update=Update.RmsProp( learningRate=1e-2, centered=True, rateDecay=0.9 ) )
   rpop = SNN_Regressor( **commonSettings, update=Update.Rprop() )

   # looks like SGD for dependent and momentum for independent

   plt.figure()
   maxIter = 1000

   plt.semilogy( train( sgde, X, y, batch=0, maxIter=maxIter, stopWait=0, triggers=commonTriggers ), label='SGD' )
   plt.semilogy( train( mome, X, y, batch=1, maxIter=maxIter, stopWait=0, triggers=commonTriggers ), label='Moment' )
   plt.semilogy( train( nest, X, y, batch=1, maxIter=maxIter, stopWait=0, triggers=commonTriggers ), label='NMoment' )
   plt.semilogy( train( adam, X, y, batch=1, maxIter=maxIter, stopWait=0, triggers=commonTriggers ), label='Adam' )
   plt.semilogy( train( rmsp, X, y, batch=1, maxIter=maxIter, stopWait=0, triggers=commonTriggers ), label='RmsProp' )
   plt.semilogy( train( rpop, X, y, batch=0, maxIter=maxIter, stopWait=0, triggers=commonTriggers ), label='RProp' )

   plt.legend()
   plt.show()