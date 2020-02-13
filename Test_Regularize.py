from DualNumber.TestLib import runSuite, nearlyEqual
from DualNumber.DualArithmetic import DualGrad as Dual
import Regularize

import numpy as np

def testCase( regType, weight, expectedLoss, expectedGrad ):
   result = regType.f( weight )
   assert nearlyEqual( result, expectedLoss )
   assert nearlyEqual( regType.df( weight ), expectedGrad )
   if isinstance( result, Dual ):
      assert nearlyEqual( result.e_, expectedGrad )

def zeroTest():
   testCase( Regularize.Zero(), Dual( [ -1, 10 ], 1 ), 0, 0 )

def l2Test():
   testCase( Regularize.Ridge( 0.1 ), Dual( [ -1, 10 ], 1 ), [ 0.1, 10 ], [ -0.2, 2 ] )

suite = []

suite.append( ( zeroTest, "Zero Regularization Test" ) )
suite.append( ( l2Test, "Ridge Regularization Test" ) )

if __name__ == "__main__":
   runSuite( suite )
