# Pierre-Alexis Roy 
# 260775494
# Phys512 - Problem Set 5 -Q2
#------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
from potential import boundCond, true_pot
from numerical import Ax, pad

# We solve the same problem as in problem 1 but now using 
# the conjugate gradient method
# Everything until the numerical computation is basically the same

#-------------------------------------------------------------------

# We define the size of our grid in n --> n x n grid
n=500
# we define the radius of our cylinder (circle)
radius = 50

# we use a function defined in potential.py to get the boundary conditions
bc, mask = boundCond(radius, n)

# when computing numerically the potential, the initial V is the boundary conditions
V=bc.copy()
V=0*bc

# we will compute a tolerance to tell the algorithm when it's converged
# Recall we are solving Ax=b or b-Ax=0
# Hence, let r be these residuals
# Ax is computed in numerical.py
b=-(bc[1:-1,0:-2]+bc[1:-1,2:]+bc[:-2,1:-1]+bc[2:,1:-1])/4.0
r = b - Ax(V, mask)


# define our tolerance (same as in problem 1)
tol = 0.01
print('Running numerical algorithm')
p=r.copy()
# We proceed to do the conjugate gradient calculation
# following the steps shown in class
for i in range(10*n):

    Ap=(Ax(pad(p),mask))
    rtr=np.sum(r*r)
    # Test convergence to our treshold tol
    if rtr <= tol:
    	print('Converged after '+str(i)+' iterations')
    	break

    # if we are not converge, update parameters
    alpha=rtr/np.sum(Ap*p)
    V=V+pad(alpha*p)
    rnew=r-alpha*Ap
    beta=np.sum(rnew*rnew)/rtr
    p=rnew+beta*p
    r=rnew


plt.imshow(V)
plt.colorbar()
plt.title('Conjugate gradient solver')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('problem2_conjgrad.pdf')
plt.close()

print('To obtain a precision of 0.01, we only needed '+str(i)+' steps')
print('whereas it took 9776 by the dumb method. This is a big win.')
