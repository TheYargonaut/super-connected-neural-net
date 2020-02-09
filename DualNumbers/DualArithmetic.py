import numpy as np

class DualNumber( object ):
    '''General class for elevating to dual numbers
    operations attempt to return type of called object to make further elevations easy'''

    def __init__( self, x=0, e=0 ):
        self.x_ = x
        self.e_ = e
    def __str__( self ):
        if self.e_ < 0:
            return "{} - {}e".format( self.x_, abs( self.e_ ) )
        return "{} + {}e".format( self.x_, self.e_ )
    
    # binary ops
    def __add__( self, other ):
        if isinstance( other, DualNumber ):
            real = self.x_ + other.x_
            inft = self.e_ + other.e_
            return type(self)( real, inft )
        return type(self)( self.x_ + other, self.e_ )
    def __sub__( self, other ):
        if isinstance( other, DualNumber ):
            real = self.x_ - other.x_
            inft = self.e_ - other.e_
            return type(self)( real, inft )
        return type(self)( self.x_ - other, self.e_ )
    def __mul__( self, other ):
        if isinstance( other, DualNumber ):
            real = self.x_ * other.x_
            inft = self.x_ * other.e_ + self.e_ * other.x_
            return type(self)( real, inft )
        return type(self)( self.x_ * other, self.e_ * other )
    def __truediv__( self, other ):
        if isinstance( other, DualNumber ):
            real =  self.x_ / other.x_
            inft = ( self.e_ * other.x_ - self.x_ * other.e_ ) / ( other.x_ ** 2 )
            return type(self)( real, inft )
        return type(self)( self.x_ / other, self.e_ / other )
    def __pow__( self, other ):
        if isinstance( other, DualNumber ):
            raise NotImplementedError
        return type(self)( self.x_ ** other, other * self.e_ * ( self.x_ ** (other - 1) ) )

    # assignment ops
    def __iadd__( self, other ):
        if isinstance( other, DualNumber ):
            self.x_ += other.x_
            self.e_ += other.e_
        self.x_ += other
    def __isub__( self, other ):
        if isinstance( other, DualNumber ):
            self.x_ -= other.x_
            self.e_ -= other.e_
        self.x_ -= other
    def __imul__( self, other ):
        if isinstance( other, DualNumber ):
            self.e_ = self.x_ * other.e_ + self.e_ * other.x_
            self.x_ *= other.x_
        self.x_ *= other
        self.e_ *= other
    def __itruediv__( self, other ):
        if isinstance( other, DualNumber ):
            self.e_ = ( self.e_ * other.x_ - self.x_ * other.e_ ) / ( other.x_ ** 2 )
            self.x_ /= other.x_
        self.x_ /= other
        self.e_ /= other
    def __ipow__( self, other ):
        if isinstance( other, DualNumber ):
            raise NotImplementedError
        self.e_ *= other * ( self.x_ ** (other - 1) )
        self.x_ **= other

    # unary ops
    def __neg__( self ):
        return type(self)( -self.x_, -self.e_ )
    def __pos__( self ):
        '''usually a NOP, treat like shallow copy'''
        return type(self)( self.x_, self.e_ )
    def __abs__( self ):
        return abs( self.x_ )
    def __invert__( self ):
        return type(self)( self.x_, -self.e_ )

    # comparison ops
    def compareTo( self, other ):
        '''equivalent to comparing the numerical type
        infinitesimal part not considered'''
        if isinstance( other, DualNumber ):
            other = other.x_
        if self.x_ > other:
            return 1
        if self.x_ == other:
            return 0
        return -1
    def __lt__( self, other ):
        return self.compareTo( other ) < 0
    def __le__( self, other ):
        return self.compareTo( other ) <= 0
    def __eq__( self, other ):
        return self.compareTo( other ) == 0
    def __ne__( self, other ):
        return self.compareTo( other ) != 0
    def __ge__( self, other ):
        return self.compareTo( other ) >= 0
    def __gt__( self, other ):
        return self.compareTo( other ) > 0

class DualNumpy( DualNumber ):
    '''Dual numbers for use with Numpy arrays'''
    def __init__( self, x=0, e=None ):
        self.x_ = np.array( x )
        if e is None:
            self.e_ = np.zeros_like( x )
        else:
            self.e_ = np.array( e )
            if not self.e_.shape:
                self.e_ = np.full( self.x_.shape, e )
        assert self.x_.shape == self.e_.shape, 'size mismatch'
    def __str__( self ):
        return "{} + ({})e".format( self.x_, self.e_ )
    
    def concatenate( self ):
        pass

class DualGrad( DualNumpy ):
    '''Dual numbers using Numpy arrays to represent multiple
    independant variables with dual numbers'''
    def __init__( self, x=0, e=None, n=1 ):
        '''n = number of independant variables to represent in e
           e will have one dimension more than x'''
        self.n_ = n
        self.x_ = np.array( x )
        if e is None:
            self.e_ = np.zeros( ( n, *self.x_.shape ) )
        else:
            self.e_ = np.array( e )
            if not self.e_.shape:
                self.e_ = np.full( ( n, *self.x_.shape ), e )

        assert ( n, *self.x_.shape ) == self.e_.shape, 'size mismatch'

    # can't really realign where e is a different size, so just assert
    
    # binary ops
    def __add__( self, other ):
        if isinstance( other, DualNumber ):
            real = self.x_ + other.x_
            inft = self.e_ + other.e_
            return DualGrad( real, inft, self.n_ )
        return DualGrad( self.x_ + other, self.e_, self.n_ )
    def __sub__( self, other ):
        if isinstance( other, DualNumber ):
            real = self.x_ - other.x_
            inft = self.e_ - other.e_
            return DualGrad( real, inft, self.n_ )
        return DualGrad( self.x_ - other, self.e_, self.n_ )
    def __mul__( self, other ):
        if isinstance( other, DualNumber ):
            real = self.x_ * other.x_
            inft = self.x_ * other.e_ + self.e_ * other.x_
            return DualGrad( real, inft, self.n_ )
        return DualGrad( self.x_ * other, self.e_ * other, self.n_ )
    def __truediv__( self, other ):
        if isinstance( other, DualNumber ):
            real = self.x_ / other.x_
            inft = ( self.e_ * other.x_ - self.x_ * other.e_ ) / ( other.x_ ** 2 )
            return DualGrad( real, inft, self.n_ )
        return DualGrad( self.x_ / other, self.e_ / other, self.n_ )
    def __pow__( self, other ):
        if isinstance( other, DualNumber ):
            raise NotImplementedError
        return DualGrad( self.x_ ** other, other * self.e_ * ( self.x_ ** (other - 1) ), self.n_ )

    # unary ops
    def __neg__( self ):
        return DualGrad( -self.x_, -self.e_, self.n_ )
    def __pos__( self ):
        '''usually a NOP, treat like shallow copy'''
        return DualGrad( self.x_, self.e_, self.n_ )
    def __abs__( self ):
        return abs( self.x_ )
    def __invert__( self ):
        return DualGrad( self.x_, -self.e_, self.n_ )

def constructParameter():
    pass