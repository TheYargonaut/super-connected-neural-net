from DualNumber.DualArithmetic import DualNumber as Dual

import numpy as np

# Model Utilites

# wrappers to interface with sklearn utilities

def formatClassTarget( y ):
   '''for when y contains integer labels 0..max'''
   yt = y.astype( int )
   Y = np.zeros( ( *y.shape, ( np.max( yt ) + 1 ) ) )
   Y[ np.arange( *y.shape ), yt ] = 1
   return Y

def formatTarget( y, expectWidth=None ):
   '''make sure y is a 2-d numpy array
   if expectLen is provided, use to check appropriate size'''
   y = np.array( y )
   if len( y.shape ) == 1:
      if expectWidth:
         y = np.reshape( y, ( -1, expectWidth ) )
      else:
         y = np.reshape( y, ( -1, 1 ) )
   if expectWidth:
      assert y.shape[ 1 ] == expectWidth
   assert len( y.shape ) == 2   
   return y

class SkRegressor( object ):
   '''wrapper class to spoof scikit-learn regressors'''
   _estimator_type = "regressor"

   def __init__( self, model, outWidth=None ):
      self.model_ = model
      self.outWidth_ = outWidth

   def fit( self, X, y ):
      '''Fit the model to data matrix X and target(s) y'''
      pass

   def partial_fit( self, X, y ):
      '''Update the model with a single iteration over the given data'''
      self.model_.partial_fit( formatTarget( y, self.outWidth_ ) )
   
   def predict( self, X ):
      '''Perform classification on samples in X'''
      raw = self.model_.predict( X )
      if isinstance( raw, Dual ):
         return raw.x_
      return raw

   def score( self, X, y, sample_weight=None ):
      '''Return the coefficient of determination R^2 of the prediction'''
      pred = self.predict( X )
      u = np.sum( ( y - pred ) ** 2 )
      v = np.sum( np.power( y, 2 ) )
      return 1 - u / v

class SkClassifier( object ):
   '''wrapper class to spoof scikit-learn classifiers'''
   _estimator_type = "classifier"

   def __init__( self, model, nClasses=None, generative=False ):
      self.model_ = model
      self.generative_ = generative
      self.nClasses_ = nClasses

   # TODO once trainer model exists
   def fit( self, X, y ):
      '''Fit the model to data matrix X and target(s) y'''
      raise NotImplementedError

   def partial_fit( self, X, y ):
      '''Update the model with a single iteration over the given data'''
      self.model_.partial_fit( formatTarget( y, self.nClasses_ ) )
   
   def predict( self, X ):
      '''Perform classification on samples in X'''
      raw = self.decision_function( X )
      if isinstance( raw, Dual ):
         raw = raw.x_
      return np.argmax( raw, axis=1 )

   def predict_log_proba( self, X ):
      '''Return the log of probability estimates'''
      return np.log( self.predict_proba( X ) )

   def predict_proba( self, X ):
      '''Probability estimates'''
      if self.generative_:
         return self.decision_function( X )
      raise NotImplementedError
   
   def decision_function( self, X):
      '''Evaluate decision function for samples in X'''
      return self.model_.predict( X )

   def score( self, X, y, sample_weight=None ):
      '''Return the mean accuracy on the given test data and labels'''
      if sample_weight is None:
         sample_weight = 1 / len( y )
      pred = self.predict( X )
      return np.sum( np.where( pred == y, np.ones_like( y ), np.zeros_like( y ) ) * sample_weight )
