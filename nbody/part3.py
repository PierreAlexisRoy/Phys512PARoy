# Pierre-Alexis Roy
# 260775494
# Phys 512 - Nbody Simulation - part 3
#----------------------------
import numpy as np 
from matplotlib import pyplot as plt
from particle import *

# Here, we let a lot of particles (in the hundred thousands) randomly
# distributed around following periodic boundary conditions (just like a thorus)


if __name__=='__main__':
    #plt.ion()
    # no initial velocity
    vi = 0
    # softening
    soft = 10
    # number of ptcls
    n=100000
    # size of our particle mesh
    grid_size = 400
    # initialize the system
    part=particles(m=1/n ,N_particles=n, grid_size=grid_size,v_init=vi, soft = soft, dt=100)
 
    # evolve it in time
    for i in range(200):
    	part.evolve()
    	plt.clf()
    	# plt.plot(part.x, part.y, '*')
    	# plt.ylim((0,grid_size))
    	# plt.xlim((0,grid_size))
    	plt.pcolormesh(part.grid)
    	plt.pause(1e-2)