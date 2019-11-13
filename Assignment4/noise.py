import numpy as np 
from scipy.ndimage import gaussian_filter


# this file holds the functions for the noise models and whitening

# define a model for the noise
def Nmodel(strain, window):
	#pass a window onto the strain
	win_strain = strain*window
	# take a norm factor to help the results
	norm = np.sqrt(np.mean(window**2))

	#get a power spectrum from the FT squared
	pow_spec = np.abs(np.fft.rfft(win_strain/norm))**2

	#gaussian smoothing
	pow_spec = gaussian_filter(pow_spec,1)

	return pow_spec


def whitening(data, noise, window):
	# return the withened strain or template using the window
	# we still need the normalization factor
	norm = np.sqrt(np.mean(window**2))
	return np.fft.rfft(window*data)/(np.sqrt(noise)*norm)


