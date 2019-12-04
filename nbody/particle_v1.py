# Pierre-Alexis Roy
# 260775494
# Phys 512 - Nbody Simulation
#----------------------------
import numpy as np 
from matplotlib import pyplot as plt


# Here, we will define a class for the Nbody simulation which 
# will include the features needed for all the parts of the problem


class particles():
    #---------------------------------------------------------------------------------------------
    def __init__(self, m=1.0, N_particles=1000, grid_size=500, v_init=0, soft=0.5, G=1.0, dt=0.1):
        '''
        initialize the needed features of our class particle

        The inputs are the following

        m           = mass of the particles,
        N_particles = number of particles in the simulation
        v           = initial velocity for the particles
        soft        = softening constant 
        G           = Newton's Constant
        dt          = Time steps 

        '''

        # Store these initial parameters in 'initial'
        self.initial              = {}
        self.initial['m']         = m
        self.initial['soft']      = soft
        self.initial['N']         = N_particles
        self.initial['grid_size'] = grid_size
        self.initial['v_init']    = v_init
        self.initial['G']         = G
        self.initial['dt']        = dt

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


    #----------------------------------------------------------------------------------
    def initial_position(self):
        '''
        get the initial positions for the particles 

        we make sure that they are within our mesh grid size

        '''
        self.x = (self.initial['grid_size'] - 1.0) * np.random.rand(self.initial['N'])
        self.y = (self.initial['grid_size'] - 1.0) * np.random.rand(self.initial['N'])
    #-----------------------------------------------------------------------------------
    def initial_velocities(self):
        '''
        get the initial velocities for the particles

        '''
        self.vx = self.initial['v_init'] * np.random.randn(self.initial['N'])
        self.vy = self.initial['v_init'] * np.random.randn(self.initial['N'])
    #-----------------------------------------------------------------------------------
    def masses(self, m):
        ''' 
        get the masses for the particles

        '''
        self.m = m*np.ones(self.initial['N'])

    # def grid(self):
    #     ''' 
    #     create a grid on which we will "put" our particles

    #     '''
    #     self.grid = np.zeros([self.initial['grid_size'], self.initial['grid_size']])

    #---------------------------------------------------------------------------------
    def particle_mesh(self):
        '''
        make our mesh grid and put the particles in it using the 
        nearest grid point

        '''

        self.grid = np.zeros([self.initial['grid_size'], self.initial['grid_size']])
        # create points where there is mass in our grid
        self.ixy=np.asarray([np.round(self.x), np.round(self.y)], dtype='int')
        
        # loop to get the mass on each grid point 
        for i in range(self.initial['N']):
            self.grid[self.ixy[0, i],self.ixy[1, i]]=self.grid[self.ixy[0, i],self.ixy[1, i]] + self.initial['m']

    #----------------------------------------------------------------------------------
    def greens_function(self):
        '''
        since we use convolutions in Fourier space, our goal is to convolute
        the green's function with the mass density rho

        Green's is equal to 1/(4PiR), we compute it on our grid

        this calculation does not depend on the masses and so we only need to do it once
        since our gridsize won't change
        '''

        # define a mesh grid using numpy
        grid_green = np.zeros([self.initial['grid_size'], self.initial['grid_size']])

        # get the size 
        n = self.initial['grid_size']

        # get softening constant squared
        soft = self.initial['soft']**2

        for x in range(len(grid_green[0])):
            for y in range(len(grid_green[1])):
                # get r squared from the point center point
                r_squared = (x-n//2)**2 + (y-n//2)**2
                # if the distance is zero, we use the softening
                if r_squared < soft:
                    r_squared = soft
                    #grid_green[x,y] = soft

                
                # add the softening for continuity  
                r_squared += soft
                # take square root
                r = np.sqrt(r_squared)
                # update the green's function
                grid_green[x,y] = 1/(4*np.pi*r)

        # we also will flip it so it is the same in each corner of our grid
        # so that it still holds when taking it to Fourier space

        # if n%2==0: # if we have an even grid size
        #     # flip the first bottom corner and flip/paste it to the other corner
        #     grid_green[n//2:, :n//2] = np.flip(grid_green[:n//2, :n//2], axis=0)
        #     # flip and maste it to the otehr half
        #     grid_green[:, n//2:] = np.flip(grid_green[:, :n//2], axis=1)

        #     # grid_green[n//2:, n//2:] = np.flip(grid_green[:n//2, :n//2])
        #     # grid_green[:n//2, n//2:] = np.flip(grid_green[:n//2, :n//2], axis=1)

        # else:# if we have an odd grid size
        #     # flip the first bottom corner and flip/paste it to the other corner
        #     grid_green[n//2:, :n//2+1] = np.flip(grid_green[:n//2+1, :n//2+1], axis=0)
        #     # flip and maste it to the otehr half
        #     grid_green[:, n//2:] = np.flip(grid_green[:, :n//2+1], axis=1)

        #     # grid_green[n//2:, n//2:] = np.flip(grid_green[:n//2+1, :n//2+1])
        #     # grid_green[:n//2+1, n//2:] = np.flip(grid_green[:n//2+1, :n//2+1], axis=1)


        # since we only need to compute it once, we put this in our init and store it
        # in self
        self.green = grid_green

    #---------------------------------------------------------------------------------
    def get_phi(self):
        '''
        We can convolute our green's function with rho
        We do this in fourier space so it becomes a multiplication
        '''
        
        # we multiply green's and rho in 2D fourier space and take the real part of
        # the inverse fourier transform
        phi = np.real(np.fft.ifft2(np.fft.fft2(self.green)*np.fft.fft2(self.grid)))

        return phi
    #----------------------------------------------------------------------------------
    def get_forces(self):
        '''
        we compute the forces on the masses by taking the gradient of
        our potential phi. We use a discretized centered difference gradient

        F = - grad(Phi)

        F_i = - (Phi_(i+1) - Phi_(i-1))/2

        '''

        # we get the potential of our grid
        phi = self.get_phi()

        # we will use a discretized gradient to compute the 
        # forces on each grid point

        forces_x = -0.5 * (np.roll(phi, 1, axis = 1) - np.roll(phi, -1, axis=1)) * self.grid
        forces_y = -0.5 * (np.roll(phi, 1, axis = 0) - np.roll(phi, -1, axis=0)) * self.grid

        return forces_x, forces_y
    #------------------------------------------------------------------------------------
    def evolve(self):
        '''
        the function which interpolates the forces on the masses on the grid
        and evolves the particles in time
        '''

        self.x += (self.vx * self.initial['dt'])
        self.x = self.x%(self.initial['grid_size']-1)
        self.y += (self.vy * self.initial['dt'])
        self.y = self.y%(self.initial['grid_size']-1)

        #print(self.x, self.y)

        # get the forces and multiply by our masses 
        fx, fy = self.get_forces() 

        # now, we compute the next velocity 
        for i in range(len(self.x)):

            self.vx[i] = fx[self.ixy[0,i], self.ixy[1,i]] * self.initial['dt']/self.initial['m']
            self.vy[i] = fy[self.ixy[0,i], self.ixy[1,i]] * self.initial['dt']/self.initial['m']


        # update the grid since the particles have moved
        self.particle_mesh()

    #----------------------------------------------------------------------------
    























# if __name__=='__main__':
#     #plt.ion()
#     n=1000
#     oversamp=5
#     part=particles(m=1.0/n,N_particles=n,dt=0.1/oversamp)
#     plt.plot(part.x,part.y,'*')
#     plt.show()

