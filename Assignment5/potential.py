import numpy as np 


# We use this file to compute the potentials

# This will create the boundary condition given a grid size n
# and the circle radius
def boundCond(radius, n):
	# need horizontal and vertical parts
	nx = np.linspace(0, n-1, n)
	ny = nx
	# We start defining the potential V and boundary conditions
	V=np.zeros([n,n])
	bc=0*V
	# Impose the circle conductor at constant V (Cylinder projection in 2d)
	# we choose a radius for our circle
	r = radius
	# we want points in the grid inside the circle of radius r 
	# to have 1 for potential. We also center the circle in our grid
	x,y = np.meshgrid(nx,ny)
	inside = (x-n/2)**2 + (y-n/2)**2 <= r**2
	bc[inside] = 1.0 # so the potential is at 1 in the conductor

	# We do a mask as in class in the difference of the circle
	mask=np.zeros([n,n],dtype='bool')
	mask[:,0]=True
	mask[:,-1]=True
	mask[0,:]=True
	mask[-1,:]=True
	mask[inside]=True

	return bc, mask

# define a function which computes the analytic potential
def true_pot(r, n, bc, mask):
	# recall the given potential V = (Constant)* log(r) + K
	# need horizontal and vertical parts
	nx = np.linspace(0, n-1, n)
	ny = nx
	x,y = np.meshgrid(nx,ny)

	# we need log(r)
	logr = np.zeros([n,n])
	#and we just take the log of every point
	e = 1e-8 # avoid log(0)
	# Apply the log of r, we can put the sqare root in front as 0.5 using log laws
	logr = 0.5*np.log(e + (x-n/2)**2 + (y-n/2)**2)

	# We also compute lambda in the constant and don't bother about the 2Pieps0 (we use 1 or 0 for V)
	# We have Vbox - Vcircle = -1 = lam (log(rbox) - log(rcircle))
	lam = -1.0/(logr[n//2, n-1] - logr[n//2, n//2 + r])

	# and we need an additive constant, again we use
	# 1 = Vcircle = lam*log(rcircle) + K
	K = 1 - lam*logr[n//2, n//2 + r]

	# we now have every thing we need to get the true potential
	V = np.zeros([n,n])
	V = lam*logr + K

	# now add the boundary conditions
	V[mask] = bc[mask]

	return V 

# This will create the boundary condition given a grid size n
# and the circle radius and also returns the grid x, y for problem 3
def bc_grid(radius, n):
	# need horizontal and vertical parts
	nx = np.linspace(0, n-1, n)
	ny = nx
	# We start defining the potential V and boundary conditions
	V=np.zeros([n,n])
	bc=0*V
	# Impose the circle conductor at constant V (Cylinder projection in 2d)
	# we choose a radius for our circle
	r = radius
	# we want points in the grid inside the circle of radius r 
	# to have 1 for potential. We also center the circle in our grid
	x,y = np.meshgrid(nx,ny)
	inside = (x-n/2)**2 + (y-n/2)**2 <= r**2
	bc[inside] = 1.0 # so the potential is at 1 in the conductor

	# We do a mask as in class in the difference of the circle
	mask=np.zeros([n,n],dtype='bool')
	mask[:,0]=True
	mask[:,-1]=True
	mask[0,:]=True
	mask[-1,:]=True
	mask[inside]=True

	return bc, mask, x, y


# we define a wire with a bump
def bumpy_bc_grid(radius, n):
	# need horizontal and vertical parts
	nx = np.linspace(0, n-1, n)
	ny = nx
	# We start defining the potential V and boundary conditions
	V=np.zeros([n,n])
	bc=0*V
	# Impose the circle conductor at constant V (Cylinder projection in 2d)
	# we choose a radius for our circle
	r = radius
	# but this time we define a bump of 10% of the diameter
	bump = 0.1*2*r
	# we want points in the grid inside the circle of radius r 
	# to have 1 for potential. We also center the circle in our grid
	x,y = np.meshgrid(nx,ny)
	inside = (x-n/2)**2 + (y-n/2)**2 <= r**2
	# put the bump on the radius on the y-axis
	inside_bump = (x-n/2)**2 + (y-n/2+r)**2 <= bump**2
	bc[inside] = 1.0 # so the potential is at 1 in the conductor
	bc[inside_bump] = 1.0

	# We do a mask as in class in the difference of the circle
	mask=np.zeros([n,n],dtype='bool')
	mask[:,0]=True
	mask[:,-1]=True
	mask[0,:]=True
	mask[-1,:]=True
	mask[inside]=True
	mask[inside_bump]=True

	return bc, mask, x, y

# we define a function which gets the electric field out of V
def E_field(V):
	# we know E = - grad(V)
	# we"ll take grad here and mind the signs when we plot it
	E = np.gradient(V)
	return E