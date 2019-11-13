import numpy as np 
from matplotlib import pyplot as plt
from noise import Nmodel, whitening
from scipy.optimize import curve_fit

# This file holds most of the functions used to perform the 
# different parts of the problem set. 

# Here is the function that will plot our noise model
def plot_noise(event, strainH, strainL, th, tl, window, i, fig, ax):
	# a few different arguments arise in the variables we giv to this function
	# i is just the loop argument, fig and ax are the the plots argument.
	# This structure allows us to plot all events in one figure and will be used
	# throughout the problem set. 

	#get the power spectrum for L and H
	powH = Nmodel(strainH, window)
	powL = Nmodel(strainL, window)

	#plot them 
	ax[i,0].semilogy(powH)
	ax[i,0].set_ylabel(event+'_Han')
	ax[i,1].semilogy(powL)
	ax[i,1].set_ylabel(event+'_Liv')
	if i==3:
		ax[i,0].set_xlabel('frequency (Hz)')
		ax[i,1].set_xlabel('frequency (Hz)')


#this functions gets the matched filters
def find_mf(event, strainH, strainL, th, tl, window):

	#get the power spectrum for L and H
	powH = Nmodel(strainH, window)
	powL = Nmodel(strainL, window)

	# we first whiten the data (A and d in our equations) using the function 
	# in noise.py
	A_white_H = whitening(th, powH, window)
	A_white_L = whitening(tl, powL, window)
	d_white_H = whitening(strainH, powH, window)
	d_white_L = whitening(strainL, powL, window)

	# apply the matched filter
	mf_H = np.fft.fftshift(np.fft.irfft(np.conj(A_white_H)*d_white_H))
	mf_L = np.fft.fftshift(np.fft.irfft(np.conj(A_white_L)*d_white_L))

	return mf_H, mf_L

def plot_mf(event, strainH, strainL, th, tl, window, Time, i, fig, ax):
	# get the matched filters to plot 
	mf_H, mf_L = find_mf(event, strainH, strainL, th, tl, window)
	#plot it
	ax[i,0].plot(Time, mf_H)
	ax[i,0].set_ylabel(event+'_Han')
	ax[i,1].plot(Time, mf_L)
	ax[i,1].set_ylabel(event+'_Liv')
	if i==3:
		ax[i,0].set_xlabel('Time (s)')
		ax[i,1].set_xlabel('Time (s)')


# this function finds the SNRs (H, L, H+L)
def SNR(event, strainH, strainL, th, tl, window):

	#need noise for L and H
	powH = Nmodel(strainH, window)
	powL = Nmodel(strainL, window)
	# we need the whitened strains
	A_white_H = whitening(th, powH, window)
	A_white_L = whitening(tl, powL, window)

	#we need the matched filters
	mf_H, mf_L = find_mf(event, strainH, strainL, th, tl, window)


	# Compute the SNRs
	SNRH = np.abs(mf_H*np.fft.fftshift(np.fft.irfft(np.sqrt(np.conj(A_white_H)*A_white_H))))
	SNRL = np.abs(mf_L*np.fft.fftshift(np.fft.irfft(np.sqrt(np.conj(A_white_L)*A_white_L))))
	total_SNR = np.sqrt(SNRH**2 + SNRL**2)

	return SNRH, SNRL, total_SNR

#plot the SNRs
def plot_SNR(event, strainH, strainL, th, tl, window, Time, i, fig, ax):
	# get the SNRs from the SNR() function
	SH, SL, Stot = SNR(event, strainH, strainL, th, tl, window)

	#plot the results
	ax[i,0].plot(Time, SH)
	ax[i,0].set_ylabel(event+'_Han')
	ax[i,1].plot(Time, SL)
	ax[i,1].set_ylabel(event+'_Liv')
	ax[i,2].plot(Time, Stot)
	ax[i,2].set_ylabel(event+'_L+H')
	if i==3:
		ax[i,0].set_xlabel('Time (s)')
		ax[i,1].set_xlabel('Time (s)')
		ax[i,2].set_xlabel('Time (s)')

# This function computes the Analytic or theoretical SNRs from the noise model
def Theo_SNR(event, strainH, strainL, th, tl, window):
	#need noise for L and H
	powH = Nmodel(strainH, window)
	powL = Nmodel(strainL, window)
	# we need the whitened strains
	A_white_H = whitening(th, powH, window)
	A_white_L = whitening(tl, powL, window)

	# We compute the analytical sugnal-to-noise juste out of our withened
	# model with the template (juste take A_white_H in real space)
	Theo_SNRH = np.abs(np.fft.irfft(A_white_H))
	Theo_SNRL = np.abs(np.fft.irfft(A_white_L))
	Theo_SNRtot = np.sqrt(Theo_SNRH**2 + Theo_SNRL**2)

	return Theo_SNRH, Theo_SNRL, Theo_SNRtot

def plot_Theo_SNR(event, strainH, strainL, th, tl, window, Time, i, fig, ax):
	# get the theoretical SNRs from the Theo_SNR() function
	SH, SL, Stot = Theo_SNR(event, strainH, strainL, th, tl, window)

	#plot the results
	ax[i,0].plot(Time, SH)
	ax[i,0].set_ylabel(event+'_Han')
	ax[i,1].plot(Time, SL)
	ax[i,1].set_ylabel(event+'_Liv')
	ax[i,2].plot(Time, Stot)
	ax[i,2].set_ylabel(event+'_L+H')
	if i==3:
		ax[i,0].set_xlabel('Time (s)')
		ax[i,1].set_xlabel('Time (s)')
		ax[i,2].set_xlabel('Time (s)')

#define a function that finds the 'middle weighted' frequency
def mid_freq(event, strainH, strainL, th, tl, window, frequency):
	#need noise for L and H
	powH = Nmodel(strainH, window)
	powL = Nmodel(strainL, window)
	# we need the whitened strains
	A_white_H = whitening(th, powH, window)
	A_white_L = whitening(tl, powL, window)

	#get the sum of the power spectra
	sum_H = np.cumsum(np.abs(A_white_H**2))
	sum_L = np.cumsum(np.abs(A_white_L**2))

	#now we get the middle weighted frequency
	mid_freqH = frequency[np.argmin(np.abs(sum_H - (np.amax(sum_H)/2)))]
	mid_freqL = frequency[np.argmin(np.abs(sum_L - (np.amax(sum_L)/2)))]

	return mid_freqH, mid_freqL


# we define a function which fits a gaussian to our SNRs
# in order to find our estimates for the time of arrival of the waves
def time_estimate(event, strainH, strainL, th, tl, window, time, func):
	# we get the SNRs from the events using our previous function
	SNRH, SNRL, dontcare = SNR(event, strainH, strainL, th, tl, window)

	# fit the curve with the func (gaussian in our case)

	# we will give an initial guess for the fit parameters
	AH = np.amax(SNRH) # the maximum peak height
	max_iH = np.argmax(SNRH) #index at which it is found
	uH = time[max_iH] # time at which the SNR is max
	sigH = 0.001 # try and error
	guessH = [uH, AH, sigH]

	AL = np.amax(SNRL) # the maximum peak height
	max_iL = np.argmax(SNRL) #index at which it is find
	uL = time[max_iL] # time at which the SNR is max
	sigL = 0.001 # try and error
	guessL = [uL, AL, sigL]

	# perform the fit with scipy 
	paramsH, covH = curve_fit(func, time[max_iH-7: max_iH+7], SNRH[max_iH-7: max_iH+7], p0=guessH)
	paramsL, covL = curve_fit(func, time[max_iL-7: max_iL+7], SNRL[max_iL-7: max_iL+7], p0=guessL)
	# we will take sigma as our errors so we only returns the best fit parameters
	return paramsH, paramsL













