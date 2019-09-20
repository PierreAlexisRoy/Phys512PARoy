# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 1 - Q4

#---------------------------------------------
import numpy as np 
from scipy import integrate
from matplotlib import pyplot as plt 
from problem3 import adapt_num_integral

# Here we do a good old E&M problem and compute an electric field 


# We set up the function we will integrate later
# The integral we need to compute is the following once having changed the variables
# we drop constants in front for simplicity
def E_func(R,z,u):
	f=(z-R*u)/((R**2 + z**2 - 2*R*z*u)**(3/2))
	return f


# we integrate this in two ways, using scipy quad and our own implementation from 
# problem 3. 

# As our conditions, we choose R=0.5 and z=[0,1] 
# (which will create a zero division)

# we integrate using our adaptative simpson's rule
def E_myInteg(a,b,tol,nmax):
	R=0.5 #radius
	z=np.linspace(0, 1, 1000) #range for z
	E=[] #create empty list for our field

	for i in z:
		f=lambda u: E_func(R,i,u) #function to integrate with u as a variable 

		# We use a try/except statement to avoid the error 
		# when there is a zero division
		try:#to avoid the crash when there is a division by zero
			result, calls=adapt_num_integral(f,a,b,tol,nmax)
		except:#just put zero instead
			result=0

		E.append(result)

	return E


# we now use scipy quad
def E_quad(a,b):
	R=0.5 #radius
	z=np.linspace(0, 1, 1000) #range for z
	E=[]

	for i in z:
		f=lambda u: E_func(R,i,u) # integrand with variable u
		#Here, we see the function from scipy does not care about the zero division
		result, error=integrate.quad(f,a,b)
		E.append(result)
	return E


# we plot the results together
zrange=np.linspace(0,1,1000) #our x axis
Result_quad=E_quad(-1,1) #the Scipy result
My_result=E_myInteg(-1,1,1e-8, 2000) # My integrator result

plt.clf()
plt.plot(zrange, Result_quad, label='scipy quad result')
plt.plot(zrange, My_result, label='my Integrator result')
plt.xlabel('height z')
plt.ylabel('Electric field')
plt.title('E field results comparison')
plt.legend()
plt.savefig('Efields.pdf')
plt.close()