# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 1 - Q1

#---------------------------------------------
import numpy as np 


# We want to compute the first derivative of exp(x) and
# exp(0.01*x) to verify our results from the analysis in Q1 
# parts a) and b). 
# We will then test it for x=0 and compare the result to 
# the real value. 

# We define a function that computes the first derivative
def firstderivative(f, x, d):
	return (8*f(x+d)-8*f(x-d)-f(x+2*d)+f(x-2*d))/(12*d)

# we define a function that computes the step size
def stepSize(f, f5, x,e):
	return np.sqrt((45*e*f(x))/(4*f5(x)))

#-----------------------------------------------------
# Case f(x)=exp(x)

# We'll try it at x=0
x=0.0
# we define our function
exp=np.exp

print('CASE : f(x)=exp(x), and we test at x=', x)

# we define the 5th derivative of f
exp5=np.exp
# a small error around numpy precision
epsilon=1e-16


# we get the stepsize and compute the derivative at 0
delta=stepSize(exp, exp5, x, epsilon)
deriv=firstderivative(exp, x, delta)

print('The result is ', deriv)
print('The error with the real value is ', np.abs(exp(x)-deriv))
print('')

#-------------------------------------------------------
# Case f(x)=exp(0.01*x)
print('CASE : f(x)=exp(0.01*x), and we test at x=', x)

# we define our function
def exp001(x):
	return np.exp(0.01*x)

# we define the 5th derivative of f
def exp0015(x):
	return (0.01)**5*exp(0.01*x)

# a small error around numpy precision
epsilon=1e-16

# we get the stepsize and compute the derivative at 0
delta=stepSize(exp001, exp0015, x, epsilon)
deriv=firstderivative(exp001, x, delta)

print('The result is ', deriv)
print('The error with the real value is ', np.abs(0.01*exp001(x)-deriv))

#----------------------------------------------------------