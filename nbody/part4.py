# Pierre-Alexis Roy
# 260775494
# Phys 512 - Nbody Simulation - part 4
#----------------------------
import numpy as np 
from matplotlib import pyplot as plt
from particle import *

# Here, we let a lot of particles following a power law for their mass distribution

# k^(-3)


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
    part=particles(m=1/n ,N_particles=n, grid_size=grid_size,v_init=vi, soft = soft, dt=10, setup='universe')
 

    # plt.plot(part.x, part.y, '*')
    # plt.ylim((0,grid_size))
    # plt.xlim((0,grid_size))
    # plt.show()

    # plt.pcolormesh(np.log10(part.grid))
    # plt.colorbar()
    # plt.show()


    #assert(1==0)
    # evolve it in time
    for i in range(200):
        part.evolve()
        plt.clf()
        # plt.plot(part.x, part.y, '*')
        # plt.ylim((0,grid_size))
        # plt.xlim((0,grid_size))
        plt.pcolormesh(np.log10(part.grid))
        plt.colorbar()

        plt.pause(1e-1)