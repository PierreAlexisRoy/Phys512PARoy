# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 1 - Q2

#---------------------------------------------
import numpy as np 
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# We will want to interpolate data from lakeshore.txt

# We first read the data from the file
f=np.loadtxt('lakeshore.txt') 
f=f.transpose()

# we get Temperature T, voltage V and 1st derivative dV/dT
T=f[0]
V=f[1]
dV=f[2]*0.001

#We want to interpolate T with values for V 

# we invert the derivative (useful later for ther error)
dT=1.0/dV

# we also switch the order of our three arrays
T=T[::-1]
V=V[::-1]
dT=dT[::-1]

# use cubic interpolation to ensure continuity 
inter=interp1d(V,T, kind='cubic', bounds_error=False, fill_value='extrapolate')

# Now, create a random array of points in the range of V
newV=np.linspace(V[0], V[-1], 500)
newT=inter(newV)


# Plot the data versus our interpolation
plt.clf()
plt.plot(V,T, 'ro',label='lakeshore data')
plt.plot(newV, newT, label='Interpolated points')
plt.xlabel('Voltage (V)')
plt.ylabel('Temperature')
plt.title('Lakeshore Interpolation')
plt.legend()
plt.savefig('LakeshoreInterp.pdf')
plt.close()

# This interpolation seems to work pretty well. 

# now we investigate our error. 

# we will use the derivative to look at our error i.e. compare 
# our numerical derivative to our interpolated function (inter)
# to the real data (given)

# we use the same numerical derivative from problem 1 
def firstderivative(f, x, d):
	return (8*f(x+d)-8*f(x-d)-f(x+2*d)+f(x-2*d))/(12*d)

# take a small stepsize
delta=1e-4


# create an array of our numerical derivatives for these points
numdev=np.zeros_like(T)
for i in range(len(V)):
	numdev[i]=firstderivative(inter, V[i], delta)

# We will find an upper bound for our error. Find the spot at which
# the error in the derivative is worst. 
errDer=np.abs(numdev-dT)
errMax=np.amax(errDer)
worst=np.where(errDer == errMax)

# we now find the closest V in our newV linspace
closest=np.abs(newV-V[worst]).argmin()


# we can now compute our error 
# we multiply the derivative value with the V difference to get 2 values
# which we compare, for the numerical derivative and data derivative. 
intError=np.abs(((newV[closest]-V[worst])*numdev[worst])-((newV[closest]-V[worst])*dT[worst]))

print('A rough estimate for an upper bound for our error (took at the worst point)')
print('Estimated error = ', intError)



