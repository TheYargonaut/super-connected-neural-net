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
    def __radd__( self, other ):
        return self + other
    def __sub__( self, other ):
        if isinstance( other, DualNumber ):
            real = self.x_ - other.x_
            inft = self.e_ - other.e_
            return type(self)( real, inft )
        return type(self)( self.x_ - other, self.e_ )
    def __rsub__( self, other ):
        if isinstance( other, DualNumber ):
            real = other.x_ - self.x_
            inft = other.e_ - self.e_
            return type(self)( real, inft )
        return type(self)( other - self.x_, -self.e_ )
    def __mul__( self, other ):
        if isinstance( other, DualNumber ):
            real = self.x_ * other.x_
            inft = self.x_ * other.e_ + self.e_ * other.x_
            return type(self)( real, inft )
        return type(self)( self.x_ * other, self.e_ * other )
    def __rmul__( self, other ):
        return self * other
    def __truediv__( self, other ):
        if isinstance( other, DualNumber ):
            real =  self.x_ / other.x_
            inft = ( self.e_ * other.x_ - self.x_ * other.e_ ) / ( other.x_ ** 2 )
            return type(self)( real, inft )
        return type(self)( self.x_ / other, self.e_ / other )
    def __rtruediv__( self, other ):
        return type(self)( other ) / self
    def __pow__( self, other ):
        if isinstance( other, DualNumber ):
            raise NotImplementedError
        inft = other * self.e_ * ( self.x_ ** (other - 1) )
        return type(self)( self.x_ ** other, inft )

    # in-place assignment operators
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
    def iexp( self ):
        raise NotImplementedError
    def isin( self ):
        raise NotImplementedError
    def icos( self ):
        raise NotImplementedError
    def itan( self ):
        raise NotImplementedError
    def ilog( self, base ):
        raise NotImplementedError

    # unary ops
    def __neg__( self ):
        return type(self)( -self.x_, -self.e_ )
    def __pos__( self ):
        '''usually a NOP, treat like shallow copy'''
        return type(self)( self.x_, self.e_ )
    def __abs__( self ):
        return type(self)( abs( self.x_ ), self.e_ if self.x_ > 0 else -self.e_ )
    def __invert__( self ):
        return type(self)( self.x_, -self.e_ )
    def exp( self ):
        real = np.exp( self.x_ )
        return type(self)( real, real * self.e_ )
    def sin( self ):
        raise NotImplementedError
    def cos( self ):
        raise NotImplementedError
    def tan( self ):
        raise NotImplementedError
    def tanh( self ):
        real = np.tanh( self.x_ )
        inft = self.e_ * ( 1 - real ** 2 )
        return type(self)( real, inft )
    def log( self ):
        real = np.log( self.x_ )
        inft = self.e_ / self.x_
        return type(self)( real, inft )

    # comparison ops; equivalent to comparing just real type
    def __lt__( self, other ):
        if isinstance( other, DualNumber ):
            return self.x_ < other.x_
        return self.x_ < other
    def __le__( self, other ):
        if isinstance( other, DualNumber ):
            return self.x_ <= other.x_
        return self.x_ <= other
    def __eq__( self, other ):
        if isinstance( other, DualNumber ):
            return self.x_ == other.x_
        return self.x_ == other
    def __ne__( self, other ):
        if isinstance( other, DualNumber ):
            return self.x_ != other.x_
        return self.x_ != other
    def __ge__( self, other ):
        if isinstance( other, DualNumber ):
            return self.x_ >= other.x_
        return self.x_ >= other
    def __gt__( self, other ):
        if isinstance( other, DualNumber ):
            return self.x_ > other.x_
        return self.x_ > other
    
    # utilities
    def where( self, condition, other ):
        '''where condition is true, substitute from other into self for return'''
        if isinstance( other, DualNumber ):
            real = other.x_ if condition else self.x_
            inft = other.e_ if condition else self.e_
            return type(self)( real, inft )
        real = other if condition else self.x_
        inft = 0 if condition else self.e_
        return type(self)( real, inft )

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
    
    # Binary ops
    def matmul( self, other ):
        if isinstance( other, DualNumpy ):
            real = np.matmul( self.x_, other.x_ )
            inft = np.matmul( other.e_, self.x_.transpose() ) + np.matmul( self.e_, other.x_ )
            return type(self)( real, inft )
        return type(self)( np.matmul( self.x_, other ), np.matmul( self.e_, other ) )

    # Unary ops
    def __abs__( self ):
        return type(self)( abs( self.x_ ), np.where( self.x_ > 0, self.e_, -self.e_ ) )
    def transpose( self ):
        return type(self)( self.x_.transpose, self.e_.transpose )
    
    # utilities
    def __getitem__( self, key ):
        ret = type(self)( 0, 0 )
        ret.x_ = self.x_[ key ]
        ret.e_ = self.e_[ key ]
        return ret
    def __setitem__( self, key, value ):
        if isinstance( value, DualNumpy ):
            self.x_[ key ] = value.x_
            self.e_[ key ] = value.e_
        self.x_[ key ] = value
        self.e_[ key ] = np.zeros_like( self.e_[ key ] )
    def where( self, condition, other ):
        '''where condition is true, substitute from other into self for return '''
        if isinstance( other, DualNumber ):
            real = np.where( condition, other.x_, self.x_ )
            inft = np.where( condition, other.e_, self.e_ )
            return type(self)( real, inft )
        real = np.where( condition, other, self.x_ )
        inft = np.where( condition, 0, self.e_ )
        return type(self)( real, inft )
    def sum( self, axis ):
        return type(self)( np.sum( self.x_, axis ), np.sum( self.e_, axis ) )
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

        # can't realign where e is a different size, so just assert
        assert ( n, *self.x_.shape ) == self.e_.shape, 'size mismatch'
    
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
    def __rtruediv__( self, other ):
        return type(self)( other, 0, self.n_ ) / self
    def __pow__( self, other ):
        if isinstance( other, DualNumber ):
            raise NotImplementedError
        return DualGrad( self.x_ ** other, other * self.e_ * ( self.x_ ** (other - 1) ), self.n_ )
    def matmul( self, other ):
        if isinstance( other, DualNumber ):
            real = np.matmul( self.x_, other.x_ )
            inft = np.matmul( other.e_, self.x_.transpose() ) + np.matmul( self.e_, other.x_ )
            return type(self)( real, inft, self.n_ )
        return type(self)( np.matmul( self.x_, other ), np.matmul( self.e_, other ), self.n_ )

    # unary ops
    def __neg__( self ):
        return DualGrad( -self.x_, -self.e_, self.n_ )
    def __pos__( self ):
        '''usually a NOP, treat like shallow copy'''
        return DualGrad( self.x_, self.e_, self.n_ )
    def __abs__( self ):
        return type(self)( abs( self.x_ ), np.where( self.x_ > 0, self.e_, -self.e_ ), self.n_ )
    def __invert__( self ):
        return DualGrad( self.x_, -self.e_, self.n_ )
    def exp( self ):
        real = np.exp( self.x_ )
        return type(self)( real, real * self.e_, self.n_ )
    def tanh( self ):
        real = np.tanh( self.x_ )
        inft = ( 1 - real ** 2 ) * self.e_
        return type(self)( real, inft, self.n_ )
    def log( self ):
        real = np.log( self.x_ )
        inft = self.e_ / self.x_
        return type(self)( real, inft, self.n_ )
    def transpose( self ):
        return type(self)( self.x_.transpose, self.e_.transpose, self.n_ )
    
    # utilities
    def __getitem__( self, key ):
        ret = type(self)( 0, 0, self.n_ )
        ret.x_ = self.x_[ key ]
        ret.e_ = self.e_[ ( slice(None), *key ) ]
        return ret
    def __setitem__( self, key, value ):
        eKey = ( slice(None), *key )
        if isinstance( value, DualGrad ):
            self.x_[ key ] = value.x_
            self.e_[ eKey ] = value.e_
            return
        self.x_[ key ] = value
        self.e_[ eKey ] = np.zeros_like( self.e_[ eKey ] )
    def sum( self, axis ):
        eAxis = axis if axis < 0 else axis + 1
        return type(self)( np.sum( self.x_, axis ), np.sum( self.e_, eAxis ), self.n_ )
    def where( self, condition, other ):
        '''where condition is true, substitute from other into self for return '''
        if isinstance( other, DualNumber ):
            real = np.where( condition, other.x_, self.x_ )
            inft = np.where( condition, other.e_, self.e_ )
            return type(self)( real, inft, self.n_ )
        real = np.where( condition, other, self.x_ )
        inft = np.where( condition, 0, self.e_ )
        return type(self)( real, inft, self.n_ )