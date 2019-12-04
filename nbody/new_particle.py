# Pierre-Alexis Roy
# 260775494
# Phys 512 - Nbody Simulation
#----------------------------
import numpy as np 
from matplotlib import pyplot as plt

class system():

    def __init__(self, m=1.0, N_particles=1000, grid_size=500, length=5, v_init=0, soft=0.5, G=1.0, dt=0.1, setup=None):

        self.initial              = {}
        self.initial['m']         = m
        self.initial['soft']      = soft
        self.initial['N']         = N_particles
        self.initial['grid_size'] = grid_size
        self.initial['v_init']    = v_init
        self.initial['G']         = G
        self.initial['dt']        = dt
        self.initial['length']    = length

        if setup == 'orbit':
            '''
            Initializing the problem for 2 masses orbiting
            '''

            # get the x and y position of the masses
            self.initial_orbit()
            # we get initial velocities which are set to v_init * random#
            self.initial_vel_orbits()


        else :
            # get the x and y position of the masses
            self.initial_position()
            # we get initial velocities which are set to v_init * random#
            self.initial_velocities()

        # get the masses for the particles
        self.masses(m)

        # define our grid
        #self.grid()

        # get green's function
        self.greens_function()

        # put our particles on the grid
        self.particle_mesh()


    def initial_position(self):

        self.x = np.abs(np.random.uniform(0, self.initial['length'], self.initial['N']))
        self.y = np.abs(np.random.uniform(0, self.initial['length'], self.initial['N']))
        self.xlim = [0, self.initial['length']]
        self.ylim = self.xlim

    def initial_velocities(self):
        self.vx = self.initial['v_init'] * np.random.randn(self.initial['N'])
        self.vy = self.initial['v_init'] * np.random.randn(self.initial['N'])

    def masses(self, m):
        self.m = m*np.ones(self.initial['N'])

    def greens_function(self):
        x_green = np.linspace(0, self.xlim[1], self.initial['grid_size'])
        y_green = 1*x_green

        grid_green = np.zeros([self.initial['grid_size'], self.initial['grid_size']])

        rsqr = (np.array([x_green,]*self.initial['grid_size']))**2 + (np.array([x_green,]*self.initial['grid_size']).transpose())**2

        rsqr[rsqr < self.initial['soft']] = self.initial['soft']
        r = np.sqrt(rsqr)

        grid_green = 1/(4*np.pi*r)

        n = self.initial['grid_size']
        grid_green[n//2:, :n//2] = np.flip(grid_green[:n//2, :n//2], axis=0)
        # flip and maste it to the otehr half
        grid_green[:, n//2:] = np.flip(grid_green[:, :n//2], axis=1)

        self.green = grid_green

    def particle_mesh(self):

        n = self.initial['grid_size']
        self.grid = np.zeros([self.initial['grid_size'], self.initial['grid_size']])
        # create points where there is mass in our grid
        self.ixy=np.asarray([np.round(self.x), np.round(self.y)], dtype='int')
        self.ixy = n * self.ixy
        self.ixy = self.ixy//self.initial['length']
        
        # loop to get the mass on each grid point 
        for i in range(self.initial['N']):
            self.grid[self.ixy[0, i],self.ixy[1, i]]=self.grid[self.ixy[0, i],self.ixy[1, i]] + self.initial['m']



