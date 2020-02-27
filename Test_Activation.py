from DualNumber.TestLib import runSuite, nearlyEqual
from DualNumber.DualArithmetic import DualGrad as Dual
import Activation
import TensorActivation as ta

import numpy as np
import tensorflow as tf

def testCase( activation, x, expectedOut, expectedGrad ):
    result = activation.f( x )
    assert nearlyEqual( result, expectedOut )
    assert nearlyEqual( activation.df( x ), expectedGrad )
    if isinstance( result, Dual ):
        assert nearlyEqual( result.e_, expectedGrad )

def identityTest():
    tv = ( Dual( [ -1, 10 ], 1 ), [ -1, 10 ], [ 1, 1 ] )
    testCase( Activation.Identity(), *tv )
    testCase( ta.Activation( ta.identity ), *tv )

def softplusTest():
    tv = ( Dual( [ -1, 10 ], 1 ), np.log( 1.0 + np.exp( [ -1, 10 ] ) ), 1 / ( 1 + np.exp( [ 1, -10 ] ) ) )
    testCase( Activation.Softplus(), *tv )
    testCase( ta.Activation( ta.softplus ), *tv )

def reluTest():
    tv = ( Dual( [ -1, 10 ], 1 ), [ 0, 10 ], [ 0, 1 ] )
    testCase( Activation.Relu(), *tv )
    testCase( ta.Activation( ta.relu ), *tv )

def leakyReluTest():
    tv = ( Dual( [ -1, 10 ], 1 ), [ -0.01, 10 ], [ 0.01, 1 ] )
    testCase( Activation.LeakyRelu(), *tv )
    testCase( ta.Activation( ta.leaky, tf.constant( 0.01 ) ), *tv )

def eluTest():
    tv = ( Dual( [ -1, 10 ], 1 ), [ np.exp( -1 ) - 1, 10 ], [ np.exp( -1 ), 1 ] )
    testCase( Activation.Elu(), *tv )
    testCase( ta.Activation( ta.elu ), *tv )

def logisticTest():
    tv = ( Dual( np.log( [ 0.1, 10 ] ), 1 ), [ 1/11, 1/1.1 ], [ 10/121, 0.1/1.21 ] )
    testCase( Activation.Logistic(), *tv )
    testCase( ta.Activation( ta.logistic ), *tv )

def tanhTest():
    tv = ( Dual( [ -1, 10 ], 1 ), np.tanh( [ -1, 10 ] ), 1 - np.tanh( [ -1, 10 ] ) ** 2 )
    testCase( Activation.Tanh(), *tv )
    testCase( ta.Activation( ta.tanh ), *tv )

def softmaxTest():
    tv = ( Dual( [ 1, 2 ], 1 ), np.exp( [ 1, 2 ] ) / np.sum( np.exp( [ 1, 2 ] ) ), [ 0, 0 ] )
    testCase( Activation.Softmax(), *tv )
    testCase( ta.Activation( ta.softmax, tf.constant( 1.0 ) ), *tv )

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
