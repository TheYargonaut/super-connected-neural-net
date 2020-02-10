from DualNumber.TestLib import runSuite
from DualNumber.DualArithmetic import DualGrad as Dual
import Activation

import numpy as np

def nearlyEqual( a, b ):
   return np.all( abs( a - b ) < 1e-6 )

def testCase( activation, x, expectedOut, expectedGrad ):
   result = activation.f( x )
   assert nearlyEqual( result, expectedOut )
   assert nearlyEqual( activation.df( x ), expectedGrad )
   if isinstance( result, Dual ):
      assert nearlyEqual( result.e_, expectedGrad )

def identityTest():
   testCase( Activation.Identity(), Dual( [ -1, 10 ], 1 ), [ -1, 10 ], [ 1, 1 ] )

def reluTest():
   testCase( Activation.Relu(), Dual( [ -1, 10 ], 1 ), [ 0, 10 ], [ 0, 1 ] )

def logisticTest():
   testCase( Activation.Logistic(), Dual( np.log( [ 0.1, 10 ] ), 1 ), [ 1/11, 1/1.1 ], [ 10/121, 0.1/1.21 ] )

def tanhTest():
   testCase( Activation.Tanh(), Dual( [ -1, 10 ], 1 ), np.tanh( [ -1, 10 ] ), 1 - np.tanh( [ -1, 10 ] ) ** 2 )

suite = []

suite.append( ( identityTest, "Identity Activation Test" ) )
suite.append( ( reluTest, "RELU Activation Test" ) )
suite.append( ( logisticTest, "Logistic Activation Test" ) )
suite.append( ( tanhTest, "Tanh Activation Test" ))

if __name__ == "__main__":
   runSuite( suite )
