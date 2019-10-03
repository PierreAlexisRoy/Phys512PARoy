# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 2 - Q1

#---------------------------------------------
import numpy as np 
from matplotlib import pyplot as plt


# The goal of this problem is to fit the base 2 logarithm with chebychev polynomials
# and to compare this to a normal polynomial least square fit. 

# we first want to fit this log2 from 0.5 to 1
a=0.5 
b=1
# Define x and y, the function we fit
x=np.linspace(a, b, 1000)
y=np.log2(x)


# Map the x linear space to (-1,1) where the Cheb fit is efficient
xx=x-x.min()
xx=xx/xx.max()
xx=xx*2-1

# We define some general variables we will use in our fit
tol=1e-6 # for truncation
order=30 # order of the polynomials
n=len(xx) 

# We now define the chebyshev polynomials and store them in A
A=np.zeros([n, order+1]) # initialize the matrix A
A[:,0]=1.0 # the 0th order Cheb polynomial
A[:,1]=xx # the 1st order Cheb poly
# we define the following ones iteratively
for i in range(1,order):
	A[:,i+1]=2*xx*A[:,i]-A[:,i-1]
# So A now holds the chebyshev polynomials


# We now fit these Chebyshev polynomials using the formula from lectures
lhs=np.dot(A.transpose(),A) #the left factor
rhs=np.dot(A.transpose(),y) #the right factor
lhs_inv=np.linalg.inv(lhs) # inverse lhs
fitp=np.dot(lhs_inv, rhs) # multiply left and right factors
# fitp now holds the fit parameters

# we now want to truncate our Chebyshev fit to keep the better coefficients
p_max=len(fitp)
for i in range(p_max):
	if np.absolute(fitp[i]) <= tol:
		p_max=i
		break
# We now have our truncated point p_max
# We can now get our final model
Cheb_model=np.dot(A[:,:p_max], fitp[:p_max])

# Plot the results
plt.clf()
plt.plot(x,y, label='true data')
plt.plot(x, Cheb_model, label='Chebyshev model')
plt.xlabel('x')
plt.ylabel('Log2(x)')
plt.legend()
plt.savefig('Cheb_fit_Q1a.pdf')
plt.close()

# we print the number of terms needed
print('We need ', p_max, 'terms to fit the function log2(x), x=[0.5,1]')

# We will now compare this to a standard polynomial fit
# compute it with numpy.polyfit to the same order p_max 
# as our Cheb_model
lin_p=np.polyfit(x,y,p_max-1) #the linear polynomials parameters
lin_model=np.polyval(lin_p, x) #our linear model

# We compare both fits by plotting the residuals
plt.clf()
plt.plot(x,Cheb_model-y, label='Chebyshev')
plt.plot(x, lin_model-y, label='Standard polynomials')
plt.xlabel('x')
plt.ylabel('Residuals')
plt.legend()
plt.savefig('Cheb_vs_Lin_Residuals.pdf')

# We compare the max errors and RMS for both fits
Cheb_err=np.max(np.abs(Cheb_model-y))
Cheb_RMS=np.sqrt(np.mean((Cheb_model-y)**2))

lin_err=np.max(np.abs(lin_model-y))
lin_RMS=np.sqrt(np.mean((lin_model-y)**2))

# We print these results

print('')
print('For the Chebyshev fit:')
print('The max error is ', Cheb_err, 'and the RMS is ', Cheb_RMS)
print('')
print('For the linear polynomial fit:')
print('The max error is ', lin_err, 'and the RMS is ', lin_RMS)
