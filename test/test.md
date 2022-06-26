# Test
## Description
This folder includes scripts for testing with pytest, where the version of pytest is the same as the one written in ../pyproject.toml

## numerical_gradient
The function calculate the gradient of Partial differential.
We test the accuracy following the method.
test_numerical_gradient define f = x^2 + y^2
Partial derivative of f with respect to x is f_x = 2^x
Partial derivative of f with respect to y is f_y = 2^y

So we calculate the partial derivative of f using numerical_gradient.
Then, calultaing the error between the caluculated function and the theorical function.
For example, if x,y=(2,3) the answer of f is f_x =4 , 6
