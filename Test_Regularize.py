from DualNumber.TestLib import runSuite, nearlyEqual
from DualNumber.DualArithmetic import DualGrad as Dual
import Regularize
import TensorRegularizer as tr

import numpy as np

def testCase( regType, weight, expectedLoss, expectedGrad ):
    result = regType.f( weight )
    assert nearlyEqual( result, expectedLoss )
    assert nearlyEqual( regType.df( weight ), expectedGrad )
    if isinstance( result, Dual ):
        assert nearlyEqual( result.e_, expectedGrad )

def zeroTest():
    tv = ( Dual( [ -1, 10 ], 1 ), 0, 0 )
    testCase( Regularize.Zero(), *tv )
    testCase( tr.Regularizer( tr.zero ), *tv )

def ridgeTest():
    tv = ( Dual( [ -1, 10 ], 1 ), [ 0.05, 5 ], [ -0.1, 1 ] )
    testCase( Regularize.Ridge( 0.1 ), *tv )
    testCase( tr.Regularizer( tr.ridge, 0.1 ), *tv )

def lassoTest():
    tv = ( Dual( [ -1, 10 ], 1 ), [ 0.05, 0.5 ], [ -0.05, 0.05 ] )
    testCase( tr.Regularizer( tr.lasso, 0.1 ), *tv )

def elasticTest():
    tv = ( Dual( [ -1, 10 ], 1 ), [ 0.1, 5.5 ], [ -0.15, 1.05 ] )
    testCase( tr.Regularizer( tr.elasticNet, 0.1, 0.1 ), *tv )

suite = []

suite.append( ( zeroTest, "Zero Regularization Test" ) )
suite.append( ( ridgeTest, "Ridge Regularization Test" ) )
suite.append( ( lassoTest, "LASSO Regularization Test" ) )
suite.append( ( elasticTest, "Elastic Net Regularization Test" ) )

if __name__ == "__main__":
   runSuite( suite )
