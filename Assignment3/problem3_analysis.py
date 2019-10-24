# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 3 - Q3 - Analysis

#---------------------------------------------
import numpy as np
import camb
from matplotlib import pyplot as plt
from wmap_camb_example import *

# Here I will study the chains made by the MCMC of problem3.py

# load the chains and chi squared
chains = np.loadtxt('chains_chi2_p3new_scale05.txt')

# we first plot chi squared --------------------------------------------------------
chi2 = chains[:,6]
# we plot chi square for all steps
plt.clf()
plt.plot(chi2)
plt.xlabel('steps')
plt.ylabel('Chi squared')
plt.savefig('problem3_chi2.pdf')
plt.close()

print('We see the chi squared has stabilized, it is a sign of convergence. We plot the chains.')
print('We also see it takes roughly 200 steps for chi square to get down so we let these steps be our burn-in')
burn=200

# plotting chains ----------------------------------------------------------------------
plt.clf() 
fig = plt.figure(figsize=(10,10))
ax1=fig.add_subplot(len(chains[0])-1, 1, 1)
plt.plot(chains[:,0])
plt.ylabel('H0')

fig.add_subplot(len(chains[0])-1, 1, 2, sharex=ax1)
plt.plot(chains[:,1])
plt.ylabel('ombh2')

fig.add_subplot(len(chains[0])-1, 1, 3, sharex=ax1)
plt.plot(chains[:,2])
plt.ylabel('omch2')

fig.add_subplot(len(chains[0])-1, 1, 4, sharex=ax1)
plt.plot(chains[:,3])
plt.ylabel('tau')

fig.add_subplot(len(chains[0])-1, 1, 5, sharex=ax1)
plt.plot(chains[:,4])
plt.ylabel('As')

fig.add_subplot(len(chains[0])-1, 1, 6, sharex=ax1)
plt.plot(chains[:,5])
plt.ylabel('ns')
plt.xlabel('steps')

fig.savefig('problem3_chains.pdf')
plt.close()

print('')
print('We see the chains seem pretty converged since when they wonder around values')
print('without showing constant growth or decay! ')


# plotting chains in fourier space to look into stability --------------------------------------------
plt.clf() 
fig = plt.figure(figsize=(10,10))
ax1=fig.add_subplot(len(chains[0])-1, 1, 1)
plt.plot(np.fft.rfft(chains[:,0]))
plt.ylabel('H0')
plt.yscale('symlog')

fig.add_subplot(len(chains[0])-1, 1, 2, sharex=ax1)
plt.plot(np.fft.rfft(chains[:,1]))
plt.ylabel('ombh2')
plt.yscale('symlog')

fig.add_subplot(len(chains[0])-1, 1, 3, sharex=ax1)
plt.plot(np.fft.rfft(chains[:,2]))
plt.ylabel('omch2')
plt.yscale('symlog')

fig.add_subplot(len(chains[0])-1, 1, 4, sharex=ax1)
plt.plot(np.fft.rfft(chains[:,3]))
plt.ylabel('tau')
plt.yscale('symlog')

fig.add_subplot(len(chains[0])-1, 1, 5, sharex=ax1)
plt.plot(np.fft.rfft(chains[:,4]))
plt.ylabel('As')
plt.yscale('symlog')

fig.add_subplot(len(chains[0])-1, 1, 6, sharex=ax1)
plt.plot(np.fft.rfft(chains[:,5]))
plt.ylabel('ns')
plt.yscale('symlog')
plt.xlabel('steps')

fig.savefig('problem3_chains_fft.pdf')
plt.close()

print('')
print('We see from the Fourier plot that the chains are really stable.')

# compute best fit parameters and error------------------------------------------------------------
fit_params=np.mean(chains[burn:,],axis=0)
error_params = np.std(chains[burn:,], axis=0)
print('')
print('The best fit parameters are: ')
print('H0                  =', fit_params[0],' +/- ', error_params[0])
print('Density baryonic    =', fit_params[1],' +/- ', error_params[1])
print('Density dark matter =', fit_params[2],' +/- ', error_params[2])
print('Tau                 =', fit_params[3],' +/- ', error_params[3])
print('Amplitude           =', fit_params[4],' +/- ', error_params[4])
print('Tilt                =', fit_params[5],' +/- ', error_params[5])
