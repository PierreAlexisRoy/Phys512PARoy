# Pierre-Alexis Roy 
# 260775494
# Phys512 - Problem Set 5 -Q5
#------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt


# For this question, we move away from electromagnetism and consider
# the Heat Equation in a box with one wall undergoing an increase in 
# temperature 

# The initial conditions for the box to be at 0 temperature

# We know the temperature will propagate perpendicular to the heating wall
# Hence, we can just solve the diffusion equation for the middle 

# Heat Equation dt(u) = k * dxx(u)
# Initial condition : u = 0
# Boundary condition : Far left is heating linearly

# We define the pararmeters we need 
# define timestep
dt = 0.0005
# size of each step of the grid
dx = 0.0005
# diffusion constant
k = 1e-5
#length of the grid (ie end of the box)
x_end = 0.025
# final time
t_end = 2
# temperature up to which we will increase the left wall
T_end = 10


# We define a solver for our diffusion equation
def diffusion_eq(dt,dx,t_end,x_end,k,T_end):
    # we need a diffusion term
    s = k*dt/dx**2
    # make axis for time and space
    x = np.arange(0,x_end+dx,dx) 
    t = np.arange(0,t_end+dt,dt)
    # initialize temperature T
    T = np.zeros([len(t),len(x)])
    #we set the final temperature
    T[:,0] = T_end
    # loopr for each time and space
    for i in range(0,len(t)-1):
        for ii in range(1,len(x)-1):
            # impose the time increasing heating of the left side
            T[i,0] = t[i]*T_end
            # solve diffusion eq.
            T[i+1,ii] = T[i,ii] + s*(T[i,ii-1] - 2*T[i,ii] + T[i,ii+1]) 
    return x,T


# we perform the algorithm
print('Solving Diffusion Equation')
x,Temp = diffusion_eq(dt,dx,t_end,x_end,k,T_end)
print('Done!')


# we plot the results
# create a time array at which we will plot T
t_array = np.arange(0,t_end,0.05)
# for each of these times, plot the temperature
for t in t_array:
    # We emphasize the times at the begining, middle and end of the process
    # with a label and a larger line
    if t==0:
        plt.plot(x,Temp[int(t//dt),:], label='time = '+str(t), linewidth=3, color='black')
    if t==0.5:
        plt.plot(x,Temp[int(t//dt),:], label='time = '+str(t), linewidth=3, color='green')
    if t==t_array[-1]:
        plt.plot(x,Temp[int(t//dt),:], label='time = '+str(t), linewidth=3, color='red')
    else:
        plt.plot(x,Temp[int(t//dt),:])

plt.title('Heat distribution along the line')
plt.xlabel('position x')
plt.ylabel('Temperature T')
plt.legend()
#plt.show()
plt.savefig('problem5_temperature.pdf')
plt.close()

print(
    '\nWhen we solve this problem, we have to set and define a number of things:\n\n'
    '- the size and time range\n'
    '- the size of timesteps and sizesteps (grid size)'
    '- the final temperature reached by the wall\n'
    '- the diffusion constant k which determines the rate of diffusion,\n'
    '  in fact, choosing too big a k ends up in the solution diverging.'
    )