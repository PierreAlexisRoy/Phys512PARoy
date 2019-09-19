# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 1 - Q4

#---------------------------------------------
import numpy as np 
from scipy import integrate
from matplotlib import pyplot as plt 
from problem3 import adapt_num_integral

# Here we do a good old E&M problem and compute an electric field 

# The integral we need to compute is the following once having changed the variables
# (1/2*eps0)*(R**2*sigma)*INTEGRAL((z-R*u)/(R**2+z**2-2*R*z*u)^(3/2))du

# we integrate this in two ways, using scipy quad and our own implementation from 
# problem 3. 

# As our conditions, we choose R=1 and z=[0,2]


# We set up the function we integrate
def E_func(R,z,u):
	f=(z-R*u)/((R**2 + z**2 - 2*R*z*u)**(3/2))
	return f

# we integrate using our adaptative simpson's rule
def E_myInteg(a,b,tol,nmax):
	R=1 #radius
	z=np.linspace(0, 2, 1000) #range for z
	E=[]

	for i in z:
		f=lambda x: E_func(R,i,x) #function to integrate

		try:#to avoid the crash when there is a division by zero
			result, calls=adapt_num_integral(f,a,b,tol,nmax)
		except:#just put zero instead
			result=0

		E.append(result)

	return E


# we now use scipy quad
def E_quad(a,b):
	R=1 #radius
	z=np.linspace(0, 2, 1000) #range for z
	E=[]

	for i in z:
		f=lambda x: E_func(R,i,x)
		#Here, we see the function from scipy does not care about the zero division
		result, error=integrate.quad(f,a,b)
		E.append(result)
	return E


# we plot the results together
plt.clf()
plt.plot(np.linspace(0,2,1000), E_quad(-1,1), label='scipy quad result')
plt.plot(np.linspace(0,2,1000), E_myInteg(-1,1,1e-8,2000), label='my algorithm result')
plt.xlabel('height z')
plt.ylabel('Electric field')
plt.title('E field results comparison')
plt.legend()
plt.savefig('Efields.pdf')
plt.close()