from DualNumber.TestLib import runSuite
from DualNumber.DualArithmetic import DualGrad as Dual
import Error

import numpy as np

def nearlyEqual( a, b ):
   return np.all( abs( a - b ) < 1e-6 )

def testCase( errorType, target, predicted, expectedError, expectedGrad ):
   result = errorType.f( target, predicted )
   assert nearlyEqual( result, expectedError )
   assert nearlyEqual( errorType.df( target, predicted ), expectedGrad )
   if isinstance( result, Dual ):
      assert nearlyEqual( result.e_, expectedGrad )


def mseTest():
   testCase( Error.Mse(), [ 1, 1 ], Dual( [ 0, 2 ], 1 ), [ 1, 1 ], [ -2, 2 ] )

def huberTest():
   testCase( Error.Huber(), [ 1, 1 ], Dual( [ 0, 2 ], 1 ), [ 0.5, 0.5 ], [ -1, 1 ] )
   testCase( Error.Huber(), [ 1, 1 ], Dual( [ 0.9, 1.1 ], 1 ), [ 0.005, 0.005 ], [ -0.1, 0.1 ] )

suite = []

suite.append( ( mseTest, "Mean Square Error Test" ) )
suite.append( ( huberTest, "Huber Error Test" ) )

if __name__ == "__main__":
   runSuite( suite )
