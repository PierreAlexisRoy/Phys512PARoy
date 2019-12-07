# Pierre-Alexis Roy
# 260775494
# Phys 512 - Nbody Simulation - part 4
#----------------------------
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.animation as animation
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
    n=200000
    # size of our particle mesh
    grid_size = 400
    # initialize the system
    part=particles(m=100/n ,N_particles=n, grid_size=grid_size,v_init=vi, soft = soft, dt=100, setup='universe')
 

    # To plot in real time -------------------------------------------------------------

    # evolve it in time
    for i in range(200):
        part.evolve()
        plt.clf()
        plt.pcolormesh(part.grid, norm=LogNorm(),cmap=plt.get_cmap('rainbow'))
        #plt.colorbar()

        plt.pause(1e-1)


    # # to create an animation ------------------------------------------------------------

    # # we will create an animation to present the results
    # # create a figure scheme
    # fig = plt.figure()
    # ax = fig.add_subplot(111, autoscale_on=False, xlim=(0,grid_size), ylim=(0,grid_size))
    # ax.set_title('Cosmology early universe')

    # # initialize our list
    # ptcl = ax.imshow(part.grid, norm=LogNorm(), cmap=plt.get_cmap('rainbow'))

    # def evolve_frame(i):
    #     global part, ax, fig
    #     # evolve
    #     part.evolve()
    #     # plot position
    #     ptcl.set_data(part.grid)
    #     return ptcl,

    # ani = animation.FuncAnimation(fig, evolve_frame, frames=200, interval=50)
    # ani.save('part4.gif', writer='imagemagick')