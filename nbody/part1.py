# Pierre-Alexis Roy
# 260775494
# Phys 512 - Nbody Simulation - part 1
#----------------------------
import numpy as np 
from matplotlib import pyplot as plt
from particle import *

# Here, we only consider one particle to verify that it remains still in the grid


if __name__=='__main__':
    #plt.ion()
    # no initial velocity
    vi = 0
    # softening
    soft = 0.01
    # number of ptcls
    n=1
    # size of our particle mesh, we use a smaller grid to be able to see the particle
    grid_size = 50
    # initialize the system
    part=particles(m=1.0,N_particles=n, grid_size=grid_size,v_init=vi, soft = soft, dt=1)

    # evolve it in time
    for i in range(200):
    	part.evolve()
    	plt.clf()
    	# plt.plot(part.x, part.y, '*')
    	# plt.ylim((0,grid_size))
    	# plt.xlim((0,grid_size))
    	plt.pcolormesh(part.grid)
    	plt.pause(1e-1)