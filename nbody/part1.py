# Pierre-Alexis Roy
# 260775494
# Phys 512 - Nbody Simulation - part 1
#----------------------------
import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.animation as animation
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


    # # to create an animation -----------------------------------------------------------

    # # we will create an animation to present the results
    # # create a figure scheme
    # fig = plt.figure()
    # ax = fig.add_subplot(111, autoscale_on=False, xlim=(0,grid_size), ylim=(0,grid_size))
    # ax.set_title('One particle')

    # # initialize our list
    # ptcl, = ax.plot([],[],'*')

    # def evovle_frame(i):
    #     global part, ax, fig
    #     # evolve
    #     part.evolve()
    #     # plot position
    #     ptcl.set_data(part.x, part.y)
    #     return ptcl,

    # ani = animation.FuncAnimation(fig, evovle_frame, frames=50)
    # ani.save('part1.gif', writer='imagemagick')


    #to plot the results in real time ----------------------------------------------

    # evolve it in time
    for i in range(100):
    	part.evolve()
    	plt.clf()
    	# plt.plot(part.x, part.y, '*')
    	# plt.ylim((0,grid_size))
    	# plt.xlim((0,grid_size))
    	plt.pause(1e-1)