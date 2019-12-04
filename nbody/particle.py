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
    def __init__(self, m=1.0, N_particles=1000, grid_size=500, v_init=0,
                 soft=0.5, G=1.0, dt=0.1, setup=None, bc = 'periodic'):
        '''
        initialize the needed features of our class particle

        The inputs are the following

        m           = mass of the particles (same for each)
        N_particles = number of particles in the simulation
        grid_size   = dimensions of our grid
        v_init      = initial velocity for the particles
        soft        = softening constant 
        G           = Newton's Constant
        dt          = Time steps 
        setup       = None       --> randomly distributed particles of equal masses
                    = 'orbit'    --> 2 particles starting in orbit at the center
                    = 'universe' --> mass fluctuations k^(-3)

        bc          = 'periodic'     --> isotropic universe (thorus geometry)
                    = 'non-periodic' --> particles near the boundaries don't feel each other 
                                         results in a universe which has a preferred position (center)

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
        self.bc                   = bc

        if setup == 'orbit':
            '''
            Initializing the problem for 2 masses orbiting at the center
            '''

            # get the x and y position of the masses in orbit
            self.initial_orbit()
            # we get initial velocities which are set to v_init 
            self.initial_vel_orbits()
            # get the masses for the particles
            self.masses(m)

        elif setup == 'universe':
            '''
            initialize the problem for part 4 with mass fluctuations
            '''
            # get the x and y position of the masses
            self.initial_position()
            # we get initial velocities which are set to v_init * random#
            self.initial_velocities()
            # we get masses for the particles which depend on their position
            self.masses_cosmo(m)


        else : 
            '''
            initialize a random isotropic problem
            '''
            # get the x and y position of the masses
            self.initial_position()
            # we get initial velocities which are set to v_init * random#
            self.initial_velocities()
            # get the masses for the particles
            self.masses(m)

        # define our grid
        #self.grid()

        # get green's function which does not change throughout the problem
        self.greens_function()

        # put our particles on the grid
        self.particle_mesh()


    #----------------------------------------------------------------------------------
    def initial_position(self):
        '''
        get the initial positions for the particles in the normal case

        we make sure that they are within our mesh grid size

        '''
        if self.bc == 'non-periodic':

            # since we will set the potential on the boundary to be zero let's add
            # a pad around the boundary where we wont put any particles
            pad = 5
            self.x = (self.initial['grid_size'] - 2*pad) * np.random.rand(self.initial['N']) + pad
            self.y = (self.initial['grid_size'] - 2*pad) * np.random.rand(self.initial['N']) + pad          

        else:
            # randomly distribute the particles
            self.x = (self.initial['grid_size'] - 1.0) * np.random.rand(self.initial['N'])
            self.y = (self.initial['grid_size'] - 1.0) * np.random.rand(self.initial['N'])
    #---------------------------------------------------------------------------------
    def initial_orbit(self):
        '''
        Initialize the position of the orbiting particles
        to be in the middle of our grid 
        '''

        # Ensure we deal with 2 particles
        self.initial['N'] = 2
        self.x = np.zeros(self.initial['N'])
        self.y = 0 * self.x

        # position our ptcls in the center
        self.x[0] = (self.initial['grid_size']/2 - 13)
        self.x[1] = (self.initial['grid_size']/2 + 13)
        self.y[0] = (self.initial['grid_size']/2)
        self.y[1] = (self.initial['grid_size']/2)

    #-----------------------------------------------------------------------------------
    def initial_velocities(self):
        '''
        get the initial velocities for the particles
        -> just v_init times a random number
        '''
        self.vx = self.initial['v_init'] * np.random.randn(self.initial['N'])
        self.vy = self.initial['v_init'] * np.random.randn(self.initial['N'])
    #-----------------------------------------------------------------------------------
    def initial_vel_orbits(self):
        '''
        define the initial velocities of the orbiting particles
        to be the same but in opposite directions
        '''

        self.vx = np.zeros(self.initial['N'])
        self.vy = 0 * self.vx
        # opposite velocities for both particles
        self.vy[0] = self.initial['v_init']
        self.vy[1] = -1 * self.vy[0]
    #-----------------------------------------------------------------------------------
    def masses(self, m):
        ''' 
        get the masses for the particles (same for each)

        '''
        self.m = m*np.ones(self.initial['N'])

    #-----------------------------------------------------------------------------------
    def masses_cosmo(self,m):
        '''
        Get the masses for the particles which will vary as r^-3 from the center
        in the case setup == 'universe'
        '''
        # initialize a mass array
        n = self.initial['grid_size']
        self.m = m * np.ones(self.initial['N'])

        # for each particle, compute radius from center and give a mass
        # accordingly
        for i in range(len(self.x)):
            rsqr = (self.x[i] - n/2)**2 + (self.y[i] - n/2)**2
            r = np.sqrt(rsqr)
            self.m[i] = m * r**(-3)
    #-----------------------------------------------------------------------------------

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
        # initialize our grid
        self.grid = np.zeros([self.initial['grid_size'], self.initial['grid_size']])
        # create points where there is mass in our grid (couples x,y for each ptcl)
        self.ixy=np.asarray([np.round(self.x), np.round(self.y)], dtype='int')
        
        # loop to get the mass (density) on each grid point by adding the associated mass
        for i in range(self.initial['N']):
            self.grid[self.ixy[0, i],self.ixy[1, i]]=self.grid[self.ixy[0, i],self.ixy[1, i]] + self.m[i]

    #----------------------------------------------------------------------------------
    def greens_function(self):
        '''
        since we use convolutions in Fourier space, our goal is to convolute
        the green's function with the mass density rho

        Green's is equal to 1/(4PiR), we compute it on our grid

        this calculation does not depend on the masses and so we only need to do it once
        since our gridsize won't change
        '''
        print('green')
        # define a mesh grid using numpy
        grid_green = np.zeros([self.initial['grid_size'], self.initial['grid_size']])

        # get the size 
        n = self.initial['grid_size']

        # get softening constant squared
        soft = self.initial['soft']**2

        for x in range(len(grid_green[0])):
            for y in range(len(grid_green[1])):
                # get r squared from the point center point
                r_squared = (x)**2 + (y)**2
                # if the distance is zero, we use the softening to avoid blow-up
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

        if n%2==0: # if we have an even grid size
            # flip the first bottom corner and flip/paste it to the other corner
            grid_green[n//2:, :n//2] = np.flip(grid_green[:n//2, :n//2], axis=0)
            # flip and maste it to the otehr half
            grid_green[:, n//2:] = np.flip(grid_green[:, :n//2], axis=1)

            # grid_green[n//2:, n//2:] = np.flip(grid_green[:n//2, :n//2])
            # grid_green[:n//2, n//2:] = np.flip(grid_green[:n//2, :n//2], axis=1)

        else:# if we have an odd grid size
            # flip the first bottom corner and flip/paste it to the other corner
            grid_green[n//2:, :n//2+1] = np.flip(grid_green[:n//2+1, :n//2+1], axis=0)
            # flip and maste it to the otehr half
            grid_green[:, n//2:] = np.flip(grid_green[:, :n//2+1], axis=1)

            # grid_green[n//2:, n//2:] = np.flip(grid_green[:n//2+1, :n//2+1])
            # grid_green[:n//2+1, n//2:] = np.flip(grid_green[:n//2+1, :n//2+1], axis=1)


        # since we only need to compute it once, we put this in our init and store it
        # in self
        self.green = grid_green

    #---------------------------------------------------------------------------------
    def get_phi(self):
        '''
        We can convolute our green's function with rho
        We do this in fourier space so it becomes a multiplication, we then
        take it back to real space 
        '''
        
        # we multiply green's and rho in 2D fourier space and take the real part of
        # the inverse fourier transform
        phi = np.real(np.fft.ifft2(np.fft.fft2(self.green)*np.fft.fft2(self.grid)))

        if self.bc == 'non-periodic':
            # we ensure the potential is centered on the particles rather than on the grid points
            phi = 0.5 * (np.roll(phi, 1, axis=1) + phi)
            phi = 0.5 * (np.roll(phi, 1, axis=0) + phi)

            # but we set potential on the boundary to 0 so our pde
            # is non-periodic at the boundaries
            phi[:, 0]   = 0
            phi[:, -1]  = 0
            phi[0, :]   = 0
            phi[-1, :]  = 0            

        else:
            # we ensure the potential is centered on the particles rather than on the grid points
            phi = 0.5 * (np.roll(phi, 1, axis=1) + phi)
            phi = 0.5 * (np.roll(phi, 1, axis=0) + phi)

        return phi
    #----------------------------------------------------------------------------------
    def get_forces(self):
        '''
        we compute the forces on the masses by taking the gradient of
        our potential phi. We use a discretized centered difference gradient

        F = - grad(Phi)

        # discretized 
        F_i = - (Phi_(i+1) - Phi_(i-1))/2

        '''

        # we get the potential of our grid
        phi = self.get_phi()

        # we will use a discretized gradient to compute the 
        # forces on each grid point and multiply by rho to get the actual forces
        forces_y = -0.5 * (np.roll(phi, 1, axis = 1) - np.roll(phi, -1, axis=1)) * self.grid
        forces_x = -0.5 * (np.roll(phi, 1, axis = 0) - np.roll(phi, -1, axis=0)) * self.grid

        return forces_x, forces_y, phi
    #------------------------------------------------------------------------------------
    def evolve(self):
        '''
        the function which interpolates the forces on the masses on the grid
        and evolves the particles in time
        '''

        # update the x and y coordinates of our particles by adding (v * dt)
        # for periodic boundary conditions, take that position modulo the size of the grid
        self.x += (self.vx * self.initial['dt'])
        self.x = self.x%(self.initial['grid_size']-1)
        self.y += (self.vy * self.initial['dt'])
        self.y = self.y%(self.initial['grid_size']-1)

        # get the forces and the potential 
        fx, fy, phi = self.get_forces() 

        # now, we compute the next velocity with F = m*a = m * dv/dt 
        for i in range(len(self.x)):
            self.vx[i] += fx[self.ixy[0,i], self.ixy[1,i]] * self.initial['dt']/self.m[i]
            self.vy[i] += fy[self.ixy[0,i], self.ixy[1,i]] * self.initial['dt']/self.m[i]

        # we compute the kinetic energy 0.5*m*v**2
        kinetic = 0.5 * np.sum(self.m * (self.vx**2 + self.vy**2))
        # we computhe the total potential
        pot = -0.5 * np.sum(phi)
        # compute the total energy
        Energy = kinetic + pot
        print('Energy is ', Energy)

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

