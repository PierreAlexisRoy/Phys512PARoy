# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 1 - Q1

#---------------------------------------------
import numpy as np 
from matplotlib import pyplot as plt


# We want to compute the first derivative of exp(x) and
# exp(0.01*x) to verify our results from the analysis in Q1 
# parts a) and b). 
# We will then test it for x=0 and compare the result to 
# the real value. 

# We define a function that computes the first derivative 
# as found in a)
def firstderivative(f, x, d):
	return (8*f(x+d)-8*f(x-d)-f(x+2*d)+f(x-2*d))/(12*d)

# we define a function that computes the step size found in a)
def stepSize(f, f5, x,e):
	return ((45*e*f(x))/(4*f5(x)))**(0.2)

#--------------------------------------------------------------------
# Case f(x)=exp(x)

# We'll try it at x=0
x=0.0
# we define our function
exp=np.exp

print('Simple Test')
print('CASE : f(x)=exp(x), and we test at x=', x)

# we define the 5th derivative of f
exp5=np.exp
# a small error around numpy precision (double)
epsilon=1e-16


# we get the stepsize and compute the derivative at 0
delta=stepSize(exp, exp5, x, epsilon)
deriv=firstderivative(exp, x, delta)

print('The result is ', deriv)
print('The error with the real value is ', np.abs(exp(x)-deriv))
print('')

#----------------------------------------------------------------------
# Case f(x)=exp(0.01*x)
print('Simple test')
print('CASE : f(x)=exp(0.01*x), and we test at x=', x)

# we define our function
def exp001(x):
	return np.exp(0.01*x)

# we define the 5th derivative of f
def exp0015(x):
	return (0.01)**5*exp(0.01*x)

# a small error around numpy precision (double)
epsilon=1e-16

# we get the stepsize and compute the derivative at 0
delta=stepSize(exp001, exp0015, x, epsilon)
deriv=firstderivative(exp001, x, delta)

print('The result is ', deriv)
print('The error with the real value is ', np.abs(0.01*exp001(x)-deriv))

#-------------------------------------------------------------------------
# Our numerical derivative seems to do a good job. 

# We now show our estimate for delta is good by plotting
# the error vs different values for delta

# For f=exp(x)

# We create an array of different deltas and compute 
# the error of our derivative for each
deltaArray=np.logspace(-14, 1, 250)

err=np.zeros_like(deltaArray)

for i in range(0,250):
	err[i]=np.abs(np.exp(x)-firstderivative(exp, x, deltaArray[i]))

# We plot our points and compare to our expected value
plt.clf()	
plt.plot(deltaArray,err, label='numerical error')
plt.axvline(stepSize(exp, exp, x, epsilon), color='r', label='best estimate')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('step size in logspace')
plt.ylabel('error of the numerical derivative')
plt.title('Error vs step size for f=exp(x)')
plt.legend()
plt.savefig('ErrorAnalysis_exp.pdf')

# For f=exp(0.01*x)
# We repeat the same for exp(0.01x)

for i in range(0,250):
	err[i]=np.abs(0.01*exp001(x)-firstderivative(exp001, x, deltaArray[i]))

plt.clf()	
plt.plot(deltaArray,err, label='numerical error')
plt.axvline(stepSize(exp001, exp0015, x, epsilon), color='r', label='best estimate')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('step size in logspace')
plt.ylabel('error of the numerical derivative')
plt.title('Error vs step size for f=exp(0.01x)')
plt.legend()
plt.savefig('ErrorAnalysis_exp001.pdf')



# We see that in both cases, our expected value is around the minimum error! 