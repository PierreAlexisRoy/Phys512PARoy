# Pierre-Alexis Roy
# 260775494
# Phys 512 - Nbody Simulation - part 3b
#----------------------------
import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from particle import *

# Here, we let a lot of particles (in the hundred thousands) randomly
# distributed around following non-periodic boundary conditions -- Not isotropic anymore


if __name__=='__main__':
    #plt.ion()
    # no initial velocity
    vi = 0
    # softening
    soft = 1
    # number of ptcls
    n=100000
    # size of our particle mesh
    grid_size = 400
    # initialize the system
    part=particles(m=1/n ,N_particles=n, grid_size=grid_size,v_init=vi, soft = soft, dt=50, bc='non-periodic')


    # # to create an animation ------------------------------------------------------------

    # # we will create an animation to present the results
    # # create a figure scheme
    # fig = plt.figure()
    # ax = fig.add_subplot(111, autoscale_on=False, xlim=(0,grid_size), ylim=(0,grid_size))
    # ax.set_title('Non-periodic BC')

    # # initialize our list
    # ptcl = ax.imshow(part.grid)

    # def evolve_frame(i):
    #     global part, ax, fig
    #     # evolve
    #     part.evolve()
    #     # plot position
    #     ptcl.set_data(part.grid)
    #     return ptcl,

    # ani = animation.FuncAnimation(fig, evolve_frame, frames=275, interval=50)
    # ani.save('part3b.gif', writer='imagemagick')



    # to plot the ptcls in real time-----------------------------------------------------

    # evolve it in time
    for i in range(200):
    	part.evolve()
    	plt.clf()
    	# plt.plot(part.x, part.y, '*')
    	# plt.ylim((0,grid_size))
    	# plt.xlim((0,grid_size))
    	plt.pcolormesh(part.grid)
    	plt.pause(1e-1)