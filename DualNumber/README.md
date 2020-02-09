# Dual Numbers for Automatic Differentiation

This library consists of 3 dual number implementations. Dual numbers make forward automatic differentiation automatic and exact using chain rule, making it especially useful for optimization tasks such as gradient-descent-based machine learning.

## Classes

### DualNumber
This simple class represents dual numbers with scalar real and infinitesimal values. It implements all basic arithmetic operations.

### DualNumpy
This class augments DualNumber to be able to represent and operate on tensors of dual numbers efficiently using Numpy.

### DualGrad
This class augments DualNumpy to be able to represent dual numbers on multiple independant variables at once, essentially turning the infinitesimal part into a gradient (set of partial derivatives).

## Tests

This library includs a series of tests which illustrates the use and capabilities of these implementations