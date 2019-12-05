# Pierre-Alexis Roy
# 260775494
# Phys 512 - Nbody Simulation - part 2
#----------------------------
import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from particle import *

# Here, we consider 2 particles that we initially put in orbit


if __name__=='__main__':
    #plt.ion()
    # initial velocity
    vi = 0.1
    # softening
    soft = 0.1
    # number of ptcls
    n=2
    # size of our particle mesh
    grid_size = 400
    # initialize the system
    part=particles(m=5,N_particles=n, grid_size=grid_size,v_init=vi, soft = soft, dt=10, setup='orbit')
 

    # # To create an animation -----------------------------------------------------------------------
    # # we will create an animation to present the results
    # # create a figure scheme
    # fig = plt.figure()
    # ax = fig.add_subplot(111, autoscale_on=False, xlim=(0,grid_size), ylim=(0,grid_size))
    # ax.set_title('Orbiting particles')

    # # initialize our list
    # ptcl, = ax.plot([],[],'*')

    # def evolve_frame(i):
    #     global part, ax, fig
    #     # evolve
    #     part.evolve()
    #     # plot position
    #     ptcl.set_data(part.x, part.y)
    #     return ptcl,

    # ani = animation.FuncAnimation(fig, evolve_frame, frames=400, interval=50)
    # ani.save('part2.gif', writer='imagemagick')

    # to plot the particles in real time --------------------------------------------

    plt.plot(part.y[0], part.x[0], '*', color='blue')
    plt.plot(part.y[1], part.x[1], '*', color='red')
    plt.ylim((0,grid_size))
    plt.xlim((0,grid_size))
    plt.pause(1)

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