# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 1 - Q2

#---------------------------------------------
import numpy as np 
from scipy.interpolate import CubicHermiteSpline 
from matplotlib import pyplot as plt

# We will want to interpolate data from lakeshore.txt

# We first read the data from the file
f=np.loadtxt('lakeshore.txt') 
f=f.transpose()

# we get voltage V, Temperature T, and 1st derivative dT
V=f[0]
T=f[1]
dT=f[2]

# Now since we have the derivative in the file, we will
# use Cubic hermite spline interpolation which will consider 
# the first derivative
CHS=CubicHermiteSpline(V,T,dT)

# Now, create a random array of points in the range of V
newV=15.2
newT=CHS(newV)

plt.clf()
plt.plot(V,T)
plt.plot(newV, newT, 'ro')
plt.show()