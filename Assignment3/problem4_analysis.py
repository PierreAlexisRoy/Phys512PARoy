# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 3 - Q4 - Analysis

#---------------------------------------------
import numpy as np
import camb
from matplotlib import pyplot as plt
from wmap_camb_example import *

# Here we study the MCMC of Question 4 (with uniform prior on Tau)
# We start by using a similar code as in problem3_analysis.py

print('')
print('Note that I used the same scaling factor of 0.5 as in problem 3')
print('Hence, the acceptance rate is a little bit lower but we still reached our minimum')
print('and in all honesty, it made the run a little faster. ')
print('')

# load the chains and chi squared
chains = np.loadtxt('chains_chi2_p4_nstep2500_scale05.txt')

# we first plot chi squared --------------------------------------------------------
chi2 = chains[:,6]
# we plot chi square for all steps
plt.clf()
plt.plot(chi2)
plt.xlabel('steps')
plt.ylabel('Chi squared')
plt.savefig('problem4_chi2.pdf')
plt.close()

print('We see the chi squared has stabilized, it is a sign of convergence. We plot the chains.')
print('We also see it takes roughly 500 steps for chi square to get down so we let these steps be our burn-in')
burn=500

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

fig.savefig('problem4_chains.pdf')
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

fig.savefig('problem4_chains_fft.pdf')
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

# using weight sampling on the chains from problem 3 -------------------------------------------------

# we define our gaussian weight
def gaussian_weight(x, u, error, N):
	return 1/(np.sqrt(2*np.pi*N)*error)*np.exp(-(x-u)**2/(2*N*error**2))

# we read our chains from problem 3 in order to importance sample them with 
# our gaussian weight
chains3 = np.loadtxt('chains_chi2_p3new_scale05.txt')

# create array to store the new weighted parameters and error
weighted_bestfit=np.zeros(6)
weighted_error=np.zeros(6)
# And now for each point of the chains, we multiply by the gaussian of
# the tau parameter with 0.0544 +/- 0.0073, with the point. 
for i in range(6):
	weight = gaussian_weight(chains3[:,3], 0.0544, 0.0073, len(chains3[:,3]))
	weighted_bestfit[i] = np.average(chains3[:,i], weights=weight)
	weighted_error[i] = np.sqrt(np.cov(chains3[:,i], aweights=weight))

#so we can now print these results for the weighted sampling
print('')
print('The results from weighted sampling the chains from problem 3 with the Planck')
print('Tau prior given we find : ')
print('H0                  =', weighted_bestfit[0],' +/- ', weighted_error[0])
print('Density baryonic    =', weighted_bestfit[1],' +/- ', weighted_error[1])
print('Density dark matter =', weighted_bestfit[2],' +/- ', weighted_error[2])
print('Tau                 =', weighted_bestfit[3],' +/- ', weighted_error[3])
print('Amplitude           =', weighted_bestfit[4],' +/- ', weighted_error[4])
print('Tilt                =', weighted_bestfit[5],' +/- ', weighted_error[5])
print('')
print('We see that indeed, we get recognizable answers for the best fit parameters')
print('We see the parameters agree within error. ')
print('Although we see that tau is a little higher since it was not constrained in our')
print('MCMC in problem3 and it spent a lot of time higher than 0.1, that is why we')
print('get a tau which is a little higher than expected. ')




