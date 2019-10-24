# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 3 - Q1

#---------------------------------------------
import numpy as np
from scipy.interpolate import interp1d
from wmap_camb_example import *

# We want to compute the Chi squared of the model from wmap_camb_example.py
# assuming gaussian uncorrelated noise. 


#we go ahead and define a function which will compute Chi-squared
def ChiSquared(model, data, noise):
	# proceed only if both the model and the data have the same length
	if len(model)==len(data):
		# compute chi squared
		chi2 = np.sum((data-model)**2/noise**2)

	else:
		print('The model and data do not have the same length!')
		exit(1)
	return chi2

# In order to be able to compute chi squared, we need to get the model 
# points at the same place as the data

# # we create an index list
# ind = np.arange(0, len(cmb), 1)
# f = interp1d(ind, cmb)
# # so f will take our camb model and find the corresponding model point
# # for each data point 

# # We define what we need to compute chi squared
# model = f(wmap[:,0])
# data = wmap[:,1]
# noise = wmap[:,2]


# we ensure the cmb model is taken at all the same points as the data
# (we shift cmb by 2 so that they match)
data = wmap[:,1]
noise = wmap[:,2]
model = cmb[2:len(data)+2]


# Compute the chi squared with our function
result = ChiSquared(model, data, noise)
print('The Chi squared for this model is', result)
print('This is close to 1588, as expected!')
