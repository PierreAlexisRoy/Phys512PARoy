# Pierre-Alexis Roy 
# 260775494
# Phys512 - Problem Set 5 -Q3
#------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
from potential import boundCond, true_pot, bc_grid
from numerical import Ax, pad
from scipy.interpolate import griddata


# Here, we want to update our solver from part 2 to apply it on 
# a range of resolutions. This is, we will solve the problem for a coarse resolution
# and interpolate that results on a more resolved box and so on. 

# So we start with a coarse resolution
n0=100
# and the desired resolution (integer multiple of n0)
res=500
# set our radius to be a tenth of our n0 so we end up with r=50 as before when we reach our resolution
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
		# also update the radius
		radius += 0.1*n0

	print('loop for resolution n = ', n)

	#reinitialize bc
	bc=0*V
	#get boundary conditions and mask and the grid for later
	bc, mask, x, y = bc_grid(radius, n) 

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
		print('We are done! It took a total of ', count, ' steps.')
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

	# we need a flattened potential, which is going to be interpolated
	V_flat = np.ravel(V)

	# We now have everything to do a linear interpolation using griddata
	# ==> we interpolate V_flat from old_grid to new_grid
	old_grid = (old_x_flat, old_y_flat)
	new_grid = (new_x_flat, new_y_flat)
	newV = griddata(old_grid, V_flat, new_grid)
	# Update potential V
	V = np.reshape(newV, (newsize, newsize))

# plot results

plt.clf()
plt.imshow(V)
plt.colorbar()
plt.title('Changing Resolution numerical scheme in '+str(count)+' steps')
plt.savefig('problem3_resolutions.pdf')
plt.close()

print('\nAgain, using this method of combining conjugate gradient with')
print('a changing resolution, we use roughly half the steps needed for part 2.')
print('Although, as we get to bigger grids, the interpolation becomes slower.')




