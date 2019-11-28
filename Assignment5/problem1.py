# Pierre-Alexis Roy 
# 260775494
# Phys512 - Problem Set 5 - Q1
#--------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
from potential import boundCond, true_pot
from numerical import Ax

# In this problem, we have a charge cylindrical conductor, held at 
# a potential V in a box with walls at potential zero. 
# We use a relaxation solver to solve for the potential everywhere in the box.

#plt.ion()

# We define the size of our grid in n --> n x n grid
n=500
# we define the radius of our cylinder (circle)
radius = 50

# we use a function defined in potential.py to get the boundary conditions
bc, mask = boundCond(radius, n)

# We also compute the true analytic potential with a function
# in potential.py
trueV = true_pot(radius, n, bc, mask)

# when computing numerically the potential, the initial V is the boundary conditions
V=bc.copy()

# we will compute a tolerance to tell the algorithm when it's converged
# Recall we are solving Ax=b or b-Ax=0
# Hence, let err be this difference
# Ax is computed in numerical.py
b=-(bc[1:-1,0:-2]+bc[1:-1,2:]+bc[:-2,1:-1]+bc[2:,1:-1])/4.0
err = b - Ax(V, mask)

# define our tolerance
tol = 0.01
print('Running numerical algorithm')
for i in range(30*n):
    V[1:-1,1:-1]=(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0
    V[mask]=bc[mask]

    # test convergence
    test = np.sum(err*err)
    if test <= tol:
    	print('Converged after '+str(i)+' iterations')
    	break
    #update the error
    err = b - Ax(V,mask)

# get the charge distribution rho as we did in class
rho=V[1:-1,1:-1]-(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0


# We will plot all results together
fig, ax=plt.subplots(1,3, figsize=(15,7))
# numerical potential
ax0 = ax[0].imshow(V)
ax[0].set_title('Numerical potential in '+str(i)+' steps')
fig.colorbar(ax0, ax=ax[0], fraction=0.046, pad=0.04)
# true potential
ax1=ax[1].imshow(trueV)
ax[1].set_title('Analytic potential')
fig.colorbar(ax1, ax=ax[1], fraction=0.046, pad=0.04)
# charge distribution
ax2=ax[2].imshow(rho)
ax[2].set_title('Charge dist.')
fig.colorbar(ax2, ax=ax[2], fraction=0.046, pad=0.04)

plt.savefig('problem1_plots.pdf')


print('\nWe see the charge density is just a circle on the edge of the wire.')
print('Indeed, in a conductor, no charge should remain inside and it should go ')
print('on the edge.')

# we compute the charge per unit length
# we will sum up our rho and divide by 2PiR to get a rough estimate
charge_tot = np.sum(rho)
charge_perl = charge_tot/(2*np.pi*radius)

print('\nWe get a charge per unit length of ', charge_perl)
print('Keep in mind that this does not take epsilon0 into account.')