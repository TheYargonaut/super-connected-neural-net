from DualNumber.TestLib import runSuite, nearlyEqual
from DualNumber.DualArithmetic import DualGrad as Dual
import Error

import numpy as np

def testCase( errorType, target, predicted, expectedError, expectedGrad ):
   result = errorType.f( target, predicted )
   assert nearlyEqual( result, expectedError )
   assert nearlyEqual( errorType.df( target, predicted ), expectedGrad )
   if isinstance( result, Dual ):
      assert nearlyEqual( result.e_, expectedGrad )

def maeTest():
   testCase( Error.Mae(), [ 1, 1 ], Dual( [ 0, 2 ], 1 ), [ 1, 1 ], [ -1, 1 ] )

def mseTest():
   testCase( Error.Mse(), [ 1, 1 ], Dual( [ 0, 2 ], 1 ), [ 1, 1 ], [ -2, 2 ] )

def huberTest():
   testCase( Error.Huber(), np.ones( 4 ), Dual( [ 0.9, 1.1, 0, 2 ], 1 ), [ 0.005, 0.005, 0.5, 0.5 ], [ -0.1, 0.1, -1, 1 ] )

def hingeTest():
   testCase( Error.Hinge(), [ 1, -1 ], Dual( [ 2, 2 ], 1 ), [ 0, 3 ], [ 0, 1 ] )

def crossEntropyTest():
   testCase( Error.CrossEntropy(), [ 1, 1 ], Dual( [ 0.5, 1 ], 1 ), [ np.log( 2 ), 0 ], [ -2, -1 ] )

def klDivergenceTest():
   testCase( Error.KlDivergence(), [ 1, 1 ], Dual( [ 0.5, 1 ], 1 ), [ np.log( 2 ), 0 ], [ -2, -1 ] )

def JsDivergenceTest():
   testCase( Error.JsDivergence(), [ 1, 1 ], Dual( [ 0.5, 1 ], 1 ), [ ( np.log( 1 / 0.75 ) + 0.5 * np.log( 0.5 / 0.75 ) ) / 2, 0 ], [ -0.20273255, 0 ] )

suite = []

suite.append( ( maeTest, "Mean Absolute Error Test" ) )
suite.append( ( mseTest, "Mean Square Error Test" ) )
suite.append( ( huberTest, "Huber Error Test" ) )
suite.append( ( hingeTest, "Hinge Error Test" ) )
suite.append( ( crossEntropyTest, "Cross Entropy Error Test" ) )
suite.append( ( klDivergenceTest, "KL-Divergence Error Test" ) )
suite.append( ( JsDivergenceTest, "JS-Divergence Error Test" ) )

if __name__ == "__main__":
   runSuite( suite )
