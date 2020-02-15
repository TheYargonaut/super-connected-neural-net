from DualNumber.TestLib import runSuite, nearlyEqual
from DualNumber.DualArithmetic import DualGrad as Dual
import Activation

import numpy as np

def testCase( activation, x, expectedOut, expectedGrad ):
   result = activation.f( x )
   assert nearlyEqual( result, expectedOut )
   assert nearlyEqual( activation.df( x ), expectedGrad )
   if isinstance( result, Dual ):
      assert nearlyEqual( result.e_, expectedGrad )

def identityTest():
   testCase( Activation.Identity(), Dual( [ -1, 10 ], 1 ), [ -1, 10 ], [ 1, 1 ] )

def softplusTest():
   testCase( Activation.Softplus(), Dual( [ -1, 10 ], 1 ), np.log( 1 + np.exp( [ -1, 10 ] ) ), 1 / ( 1 + np.exp( [ 1, -10 ] ) ) )

def reluTest():
   testCase( Activation.Relu(), Dual( [ -1, 10 ], 1 ), [ 0, 10 ], [ 0, 1 ] )

def leakyReluTest():
   testCase( Activation.LeakyRelu(), Dual( [ -1, 10 ], 1 ), [ -0.01, 10 ], [ 0.01, 1 ] )

def eluTest():
   testCase( Activation.Elu(), Dual( [ -1, 10 ], 1 ), [ np.exp( -1 ) - 1, 10 ], [ np.exp( -1 ), 1 ] )

def logisticTest():
   testCase( Activation.Logistic(), Dual( np.log( [ 0.1, 10 ] ), 1 ), [ 1/11, 1/1.1 ], [ 10/121, 0.1/1.21 ] )

def tanhTest():
   testCase( Activation.Tanh(), Dual( [ -1, 10 ], 1 ), np.tanh( [ -1, 10 ] ), 1 - np.tanh( [ -1, 10 ] ) ** 2 )

def softmaxTest():
   testCase( Activation.Softmax(), Dual( [ 1, 2 ], 1 ), np.exp( [ 1, 2 ] ) / np.sum( np.exp( [ 1, 2 ] ) ), [ 0, 0 ] )

suite = []

suite.append( ( identityTest, "Identity Activation Test" ) )
suite.append( ( softplusTest, "SoftPlus Activation Test" ) )
suite.append( ( reluTest, "RELU Activation Test" ) )
suite.append( ( leakyReluTest, "Leaky RELU Activation Test" ) )
suite.append( ( eluTest, "ELU Activation Test" ) )
suite.append( ( logisticTest, "Logistic Activation Test" ) )
suite.append( ( tanhTest, "Tanh Activation Test" ))
suite.append( ( softmaxTest, "SoftMax Activation Test" ))

if __name__ == "__main__":
   runSuite( suite )
