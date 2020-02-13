import DualArithmetic as DA
from TestLib import runSuite
import pdb
import numpy as np

def dualScalarTest( DualType ):
    '''all types of dual number should be able to pass'''
    
    # declaration
    a = DualType( 10, 2 )
    b = DualType( -1, 1 )
    c = DualType( -1, -1 )

    # comparisons
    assert a > b
    assert a == 10
    assert b == c

    # binary ops
    y = a + b
    assert y == 9
    assert y.e_ == 3
    y = a * b
    assert y == -10
    assert y.e_ == 8
    y = abs( b )
    assert y == 1
    assert y.e_ == -1
    y = a.where( b < 1, b )
    assert y == b

def dualTensorTest( DualType ):
    '''dual numbers optimized for tensors'''
    
    # declarations
    a = DualType( [ 1, 10, 5 ], 1 )
    b = DualType( [ [ 1, 2, -3 ],
                    [ 4, -5, 6 ],
                    [ -7, 8, 9 ] ], 1 )
    c = DualType( [ 5, -10, 20 ], 1 )

    # comparisons
    assert np.all( a.x_ == [ 1, 10, 5 ] )
    
    # binary ops
    y = a + c
    assert np.all( y.x_ == [ 6, 0, 25 ] )
    assert np.all( y.e_ == [ 2, 2, 2 ] )
    y = a * b
    assert np.all( y.x_ == [ [ 1, 20, -15 ],
                             [ 4, -50, 30 ],
                             [ -7, 80, 45 ] ] )
    assert np.all( y.e_ == [ [  2, 12,  2 ],
                             [  5,  5, 11 ],
                             [ -6, 18, 14 ] ] )
    y = abs( c )
    assert np.all( y == [ 5, 10, 20 ] )
    assert np.all( y.e_ == [ 1, -1, 1 ] )
    y = a.matmul( b )
    assert np.all( y == [ 6, -8, 102 ] )
    assert np.all( y.e_ == [ 14, 21, 28 ] )

def dualGradientTest( DualType ):
    '''dual numbers for multiple independant variables'''
    
    # declarations
    a = DualType( 5, 1, 4 )
    b = DualType( 4, [ 1, 2, 3, 4 ], 4 )
    c = DualType( [ 1, 10, 5 ], [[ 1, 2, 3 ]]*4, 4 )
    d = DualType( [ [ 1, 2, -3 ],
                    [ 4, -5, 6 ],
                    [ -7, 8, 9 ] ], 1, 4 )

    # comparisons
    assert np.all( a.x_ == 5 )

    # binary ops
    y = a + b
    assert np.all( y.x_ == 9 )
    assert np.all( y.e_ == [ 2, 3, 4, 5 ] )
    y = a * b
    assert np.all( y.x_ == 20 )
    assert np.all( y.e_ == [ 9, 14, 19, 24 ] )
    y = c.matmul( d )
    assert np.all( y == [ 6, -8, 102 ] )
    assert np.all( y.e_ == [[ 4, 32, 52 ]]*4 )

suite = []

suite.append( ( lambda: dualScalarTest( DA.DualNumber ), "DualNumber Scalar Test" ) )
suite.append( ( lambda: dualScalarTest( DA.DualNumpy ), "DualNumpy Scalar Test" ) )
suite.append( ( lambda: dualScalarTest( DA.DualGrad ), "DualGrad Scalar Test" ) )

suite.append( ( lambda: dualTensorTest( DA.DualNumpy ), "DualNumpy Tensor Test" ) )
suite.append( ( lambda: dualTensorTest( DA.DualGrad ), "DualGrad Tensor Test" ) )

suite.append( ( lambda: dualGradientTest( DA.DualGrad ), "DualGrad Gradient Test" ) )

if __name__ == "__main__":
   runSuite( suite )
