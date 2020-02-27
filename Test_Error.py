from DualNumber.TestLib import runSuite, nearlyEqual
from DualNumber.DualArithmetic import DualGrad as Dual
import Error
import TensorLoss as tl

import numpy as np
import pdb

def testCase( errorType, target, predicted, expectedError, expectedGrad ):
    result = errorType.f( target, predicted )
    assert nearlyEqual( result, expectedError )
    assert nearlyEqual( errorType.df( target, predicted ), expectedGrad )
    if isinstance( result, Dual ):
        assert nearlyEqual( result.e_, expectedGrad )

def maeTest():
    tv = ( np.ones( 2, dtype=np.float32 ), Dual( [ 0, 2 ], 1 ), [ 1, 1 ], [ -1, 1 ] )
    testCase( Error.Mae(), *tv )
    testCase( tl.Loss( tl.mae ), *tv )

def mseTest():
    tv = ( np.ones( 2, dtype=np.float32 ), Dual( [ 0, 2 ], 1 ), [ 1, 1 ], [ -2, 2 ] )
    testCase( Error.Mse(), *tv )
    testCase( tl.Loss( tl.mse ), *tv )

def huberTest():
    tv = ( np.ones( 4, dtype=np.float32 ), Dual( [ 0.9, 1.1, 0, 2 ], 1 ), [ 0.005, 0.005, 0.5, 0.5 ], [ -0.1, 0.1, -1, 1 ] )
    testCase( Error.Huber(), *tv )
    testCase( tl.Loss( tl.huber ), *tv )

def hingeTest():
    tv = ( np.array( [ 1, -1 ], dtype=np.float32 ), Dual( [ 2, 2 ], 1 ), [ 0, 3 ], [ 0, 1 ] )
    testCase( Error.Hinge(), *tv )
    testCase( tl.Loss( tl.hinge ), *tv )

def crossEntropyTest():
    tv = ( np.ones( 2, dtype=np.float32 ), Dual( [ 0.5, 1 ], 1 ), [ np.log( 2 ), 0 ], [ -2, -1 ] )
    testCase( Error.CrossEntropy(), *tv )
    testCase( tl.Loss( tl.crossEntropy ), *tv )

def klDivergenceTest():
    tv = ( np.ones( 2, dtype=np.float32 ), Dual( [ 0.5, 1 ], 1 ), [ np.log( 2 ), 0 ], [ -2, -1 ] )
    testCase( Error.KlDivergence(), *tv )
    testCase( tl.Loss( tl.klDivergence ), *tv )

def JsDivergenceTest():
    tv = ( np.ones( 2, dtype=np.float32 ), Dual( [ 0.5, 1 ], 1 ), [ ( np.log( 1 / 0.75 ) + 0.5 * np.log( 0.5 / 0.75 ) ) / 2, 0 ], [ -0.20273255, 0 ] )
    testCase( Error.JsDivergence(), *tv )
    testCase( tl.Loss( tl.jsDivergence ), *tv )

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
