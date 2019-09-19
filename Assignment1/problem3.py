# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 1 - Q3

#---------------------------------------------
import numpy as np 

# We want to write a smart adaptative integral algorithm


# We first implement the simpson's rule fwhich return
# the mid interval function evaluations 
def simpson_half(f, a, b, fa, fb):
    # f is function, (a,b) is the interval on which we integrate
    # fa and fb are f evaluated at a and b. 
    if a == b:
        return 0    

    # get the function evaluated at the middle of the interval
    mid=(a+b)/2
    fmid=f(mid)

    #simpson's rule
    integ=((b-a)/6)*(fa+4*fmid+fb)

    #return the result and the new midpoint and f(midpoint)
    return integ, mid, fmid


#This is the adaptative simpson rule. In which we will iteratively split 
# our interval in two to get better precision. 
def adapt_simpson(f, a, b, fa, fb, tol, last, m, fm, nmax, calls=0):
    # f is function, (a,b) is the interval on which we integrate
    # fa and fb are f evaluated at a and b. 
    # tol is the error we accept, last is the last result from
    # simpson's rule, m and fm are the previously returned midpoints
    # and function evaluation at that point, nmax is the max number of iteration 
    # we allow, and calls is the number of times simpson's rule is called.

    if calls < nmax:
        # split interval in two parts
        left, lm, flm=simpson_half(f, a, m, fa, fm)
        right, rm, frm=simpson_half(f, m, b, fm, fb)

        # compute error with last evaluation
        err=np.abs(left+right-last)

        if err<15*tol: #Terminate is error is small enough
            return last, calls
        else:#Iterate (ie split interval in two again)
            lpart=adapt_simpson(f,a,m,fa,fm,tol/2.0, left, lm, flm, nmax, calls+1)
            rpart=adapt_simpson(f,m,b,fm,fb, tol/2.0, right, rm, frm, nmax, calls+1)
            return lpart+rpart

# This is where the integral is computed, calling the previously defined
# functions. 
def adapt_num_integral(f, a, b, tol, nmax):
    # f is the function, (a,b) is the interval, tol is the error acceptance
    # and nmax is the max number of iterations allowed. 
    fa=f(a)
    fb=f(b)
    integ, m, fm=simpson_half(f,a,b,fa,fb)
    NumInteg=adapt_simpson(f,a,b,fa,fb,tol,integ,m, fm, nmax)
    # In order to get the result and number of function calls, we
    # must add up the array from adapt_simpson()
    NumInteg=np.array(NumInteg)
    calls=np.sum(NumInteg[1::2])
    result=np.sum(NumInteg[0::2])
    return result, calls

#-------------------------------------------------------------------------------
# we try the algorithm for a few functions

# f(x)=sin(x) from 0 to Pi -----------------------------------
result=adapt_num_integral(np.sin, 0, np.pi, 1e-8, 1000)
# we know the real result is 2
error=np.abs(2-result[0])
print('')
print('Test of the adaptative integral algorithm for f=sin(x)')
print('The result is', result[0],'achieved in', result[1], 'function calls')
print('The error with the true value is ', error)

#f(x)=exp(x) from 0 to 1------------------------------------
result=adapt_num_integral(np.exp, 0, 1.0, 1e-8, 1000)
# we know the real result is exp(1)-1
error=np.abs((np.exp(1)-1)-result[0])
print('')
print('Test of the adaptative integral algorithm for f=exp(x)')
print('The result is', result[0],'achieved in', result[1], 'function calls')
print('The error with the true value is ', error)

# f(x)=lorentzian from -3 to 3---------------------------------------
def funk(x):
    return 1.0/(1.0+x**2)
result=adapt_num_integral(funk, -3, 3, 1e-8, 3000)

print('')
print('Test of the adaptative integral algorithm for the Lorentzian function')
print('The result is', result[0],'achieved in', result[1], 'function calls')
#-----------------------------------------------------------------------
# It seems everything works out as it is supposed to. 

print('')
print('')
print('In all cases, we usually need 1/3 the steps needed with the lazy way')
print('from class since we do not repeat middle function evaluations')