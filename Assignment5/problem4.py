# Pierre-Alexis Roy 
# 260775494
# Phys512 - Problem Set 5 -Q4
#------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
from potential import boundCond, true_pot, bc_grid, bumpy_bc_grid, E_field
from numerical import Ax, pad
from scipy.interpolate import griddata


# in this part, we repeat a similar task as in part 3 but we add
# a little bump in our wire. Hence, I use the exact same code as in my problem 3 
# but now, i use a new function for the bumpy boundary conditions defined
# in potential.py

# we will use a constant linear increase of our resolution
# ie. start with n0 x n0 and add n0 x n0 on each iteration
# until we reach a desired resolution. 
# In our case, we want to reach 500 x 500 and we will start from 100 x 100


# So we start with a coarse resolution
n0=100
# and the desired resolution
res=500
# set our radius to be a tenth of our n0 so we end up with r=50 as before 
radius = 10
# set our tolerance
tol=0.01
# initialize, V, bc, and a counter
V=np.zeros([n0,n0])
bc=0*V
count=0

#initiate the loop for each resolution
for i in range(6):

	if i==0: # start at n0
		n = n0
	else: # add n0 at each subsequent step
		n += n0	
		radius += 0.1*n0

	print('loop for resolution n = ', n)

	#reinitialize bc
	bc=0*V
	#get boundary conditions and mask and the grid for later
	# this function puts a bump --> defined in potential.py
	bc, mask, x, y = bumpy_bc_grid(radius, n) 

	# Now proceed as in problem 2 with conjugate gradient
	b=-(bc[1:-1,0:-2]+bc[1:-1,2:]+bc[:-2,1:-1]+bc[2:,1:-1])/4.0
	r = b - Ax(V, mask)
	p=r.copy()
	# We proceed to do the conjugate gradient calculation
	# following the steps shown in class
	for ii in range(5*n):

	    Ap=(Ax(pad(p),mask))
	    rtr=np.sum(r*r)

	    # Test convergence to our treshold tol
	    if rtr <= tol: # we are converged
	    	print('Converged after '+str(ii)+' iterations\n')
	    	break

	    # if we are not converge, update parameters of conj grad
	    alpha=rtr/np.sum(Ap*p)
	    V=V+pad(alpha*p)
	    rnew=r-alpha*Ap
	    beta=np.sum(rnew*rnew)/rtr
	    p=rnew+beta*p
	    r=rnew

	# Check if we met our resolution
	if n==res:
		print('We are done!')
		break
	# update all parameters for next resolution
	# update the total step count
	count += ii

	# get the size of the next upper resolution grid
	newsize = n + n0 
	# get new dimensions and a new meschgrid
	new_nx = np.linspace(0, n-1, newsize)
	new_ny = new_nx
	new_x, new_y = np.meshgrid(new_nx, new_ny)

	# we flatten this new grid to be able to use griddata from scipy
	new_x_flat = np.ravel(new_x)
	new_y_flat = np.ravel(new_y)

	# we will need to get the old flattened grid
	old_x_flat = np.ravel(x)
	old_y_flat = np.ravel(y)

	# we need a flattened potential, that's what is going to be interpolated
	V_flat = np.ravel(V)

	# We now have everything to do a linear interpolation using griddata
	# ==> We interpolate V_flat from old_grid to new_grid
	old_grid = (old_x_flat, old_y_flat)
	new_grid = (new_x_flat, new_y_flat)
	newV = griddata(old_grid, V_flat, new_grid)
	# Update potential V
	V = np.reshape(newV, (newsize, newsize))


# we want to study the Electric field
# E_field() is in potential.py and just takes the gradient of V
# We will use a quiver plot to plot arrows for the E-field, so
# we compute E not at every point in V (o.w. we get a black image)
# we will take one in every k point
k=8
V_lowres = V[::k, ::k]
E = E_field(V_lowres)

# Plot the results
#plt.clf()
# the obtained potential
fig, ax =plt.subplots(1,2, figsize=(10, 7))
ax0 = ax[0].imshow(V)
ax[0].set_title('Bumpy V reached in '+str(count)+' steps')
fig.colorbar(ax0, ax=ax[0], fraction=0.046, pad=0.04)
# the potential along with E-field
ax1 = ax[1].imshow(V)
ax[1].quiver(x[::k, ::k],y[::k, ::k],-E[1], E[0])
ax[1].set_title('V and E-Field')
fig.colorbar(ax1, ax=ax[1], fraction=0.046, pad=0.04)
#plt.show()
plt.savefig('problem4_bumpEV.pdf')
plt.close()

print('\nThe Efield is greater around the bump than at any place around the wire.')

