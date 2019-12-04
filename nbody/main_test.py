# Pierre-Alexis Roy
# 260775494
# Phys 512 - Nbody Simulation
#----------------------------
import numpy as np 
from matplotlib import pyplot as plt

from particle import *

#plt.ion()

if __name__=='__main__':
    #plt.ion()
    vi = 0
    soft = 0.1
    n=2
    grid_size = 32
    oversamp=5
    part=particles(m=1.0,N_particles=n, grid_size=grid_size,v_init=vi, soft = soft, dt=1)
 


    plt.pcolormesh(part.green)
    plt.colorbar()
    plt.show()

    part.particle_mesh()
    #plt.plot(part.x,part.y,'*')
    plt.pcolormesh(part.grid)
    plt.colorbar()
    plt.show()

    phi = part.get_phi()
    plt.pcolormesh(phi)
    plt.colorbar()
    plt.show()



    # phi = part.get_phi()
    # print('dimension of phi = ', phi.shape)
    # phi[:,-1] = 0
    # phi[:, 0] = 0
    # phi[0, :] = 0
    # phi[-1,:] = 0
    # plt.clf()
    # plt.pcolormesh(phi)
    # plt.colorbar()
    # plt.show()

    # fx, fy = part.get_forces()
    # print(fx[part.ixy[0,0], part.ixy[1,0]])
    # print('dimension of fx = ', fx.shape)
    # print('dimension of fy = ', fy.shape)
    # plt.pcolormesh(fx)
    # plt.colorbar()
    # plt.show()



    #test evolve

    # for i in range(200):
    # 	part.evolve()
    # 	plt.clf()
    # 	# plt.plot(part.x, part.y, '*')
    # 	# plt.ylim((0,grid_size))
    # 	# plt.xlim((0,grid_size))
    # 	plt.pcolormesh(part.grid)
    # 	plt.pause(1e-1)


    



