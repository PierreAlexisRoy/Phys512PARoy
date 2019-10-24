# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 3 - Q3

#---------------------------------------------
import numpy as np
import camb
from matplotlib import pyplot as plt
from wmap_camb_example import *

# In this problem, we will fit the data using a MCMC algorithm with 
# the 6 parameters of the CAMB model. 

# We will use the covariance matrix from problem 2 
cov_mat = np.array([[ 1.34450312e+01,  2.57312553e-03, -2.39789117e-02,
         4.14038608e-01,  1.57966560e-09,  8.47753631e-02],
       [ 2.57312553e-03,  7.24538230e-07, -3.70600588e-06,
         9.82778830e-05,  3.91550609e-13,  2.08179053e-05],
       [-2.39789117e-02, -3.70600588e-06,  4.95027134e-05,
        -7.04173289e-04, -2.60502481e-12, -1.36512047e-04],
       [ 4.14038608e-01,  9.82778830e-05, -7.04173289e-04,
         2.23170015e-02,  8.84943228e-11,  3.34482645e-03],
       [ 1.57966560e-09,  3.91550609e-13, -2.60502481e-12,
         8.84943228e-11,  3.52430486e-19,  1.32130098e-11],
       [ 8.47753631e-02,  2.08179053e-05, -1.36512047e-04,
         3.34482645e-03,  1.32130098e-11,  6.85759157e-04]])


# we define the function which will take a step using the covariance matrix
def take_step_cov(covmat):
    mychol=np.linalg.cholesky(covmat)
    return np.dot(mychol,np.random.randn(covmat.shape[0]))

# we open a file in which we will store the chains
f = open('chains_chi2_p3new_scale05.txt', 'w')
# we define our data
data = wmap[:,1]
# we define the noise
noise = wmap[:,2]
# define number of step max
nstep = 2500
# we define our initial guess -- the same as wmap_example except for H0 higher 
# since I found that there is some kind of local min around 63 and don't want 
# my chains to take forever in order to get out of it. 
pars=np.asarray([71,0.02,0.1,0.07,2e-9,0.96])
params = pars
# We go on and write the MCMC algorithm adapting the one we did in class
npar=len(pars)
# we create array to store chains
chains=np.zeros([nstep,npar])
chisq=np.sum( (data-get_spectrum(pars)[2:len(data)+2])**2/noise**2)
scale_fac=0.5
chisqvec=np.zeros(nstep)
accept = 0 # used to study the acceptance rate
for i in range(nstep):
    new_params=params+take_step_cov(cov_mat)*scale_fac
    if new_params[3] > 0:

        new_model=get_spectrum(new_params)[2:len(data)+2]
        new_chisq=np.sum( (data-new_model)**2/noise**2)
    
        delta_chisq=new_chisq-chisq
        prob=np.exp(-0.5*delta_chisq)
        accept_prob=np.random.rand(1)<prob
        if accept_prob:
            accept += 1
            params=new_params
            model=new_model
            chisq=new_chisq
    chains[i,:]=params
    chisqvec[i]=chisq
    # store the chains and Chi2 in a txt file
    for ii in params:
        f.write(f'{ii} ')
    f.write(f'{chisq}\n')
    f.flush()

f.close()  
print('acceptance rate is', accept/nstep)  
fit_params=np.mean(chains,axis=0)