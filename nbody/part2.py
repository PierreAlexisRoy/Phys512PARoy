# Pierre-Alexis Roy
# 260775494
# Phys 512 - Nbody Simulation - part 2
#----------------------------
import numpy as np 
from matplotlib import pyplot as plt
from particle import *

# Here, we only consider one particle to verify that it remains still in the grid


if __name__=='__main__':
    #plt.ion()
    # no initial velocity
    vi = 0.1
    # softening
    soft = 0.1
    # number of ptcls
    n=2
    # size of our particle mesh, we use a smaller grid to be able to see the particle
    grid_size = 400
    # initialize the system
    part=particles(m=5,N_particles=n, grid_size=grid_size,v_init=vi, soft = soft, dt=10, setup='orbit')
 

    plt.plot(part.y[0], part.x[0], '*', color='blue')
    plt.plot(part.y[1], part.x[1], '*', color='red')
    plt.ylim((0,grid_size))
    plt.xlim((0,grid_size))
    plt.pause(1)
    # plt.show()
    # assert(1==0)
    # evolve it in time
    for i in range(1000):
        part.evolve()
        plt.clf()
        plt.plot(part.y[0], part.x[0], '*', color='blue')
        plt.plot(part.y[1], part.x[1], '*', color='red')
        plt.ylim((0,grid_size))
        plt.xlim((0,grid_size))
        # plt.pcolormesh(part.grid)
        # plt.colorbar()
        plt.pause(1e-2)