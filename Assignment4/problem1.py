# Pierre-Alexis Roy
# 260775494
# Phys 512 - Assignment 4
#----------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
#import my functions to read the files
from read_ligo import read_template, read_file
#import models from noise.py
from noise import Nmodel
from wave_finder import plot_noise, find_mf, plot_mf, SNR, plot_SNR, Theo_SNR, plot_Theo_SNR, mid_freq, time_estimate

# In this Assignment we will read Ligo data and find gravitational waves
# This file can be considered like our "main" in that it will make use
# of functions found in read_ligo.py, noise.py and find_waves.py


#we open a file for writing our answers and results
f = open('Results.txt', 'w')
f.write('Pierre-Alexis Roy \n')
f.write('260775494\n\n')
f.write('This file holds my results and main explanations/answers to the different\n')
f.write('parts of this problem set. \n\n')
#-------------------------------------------------------------------------------------------------------------------
print('Reading ...')
# first we create a variable for the path to the data
loc = 'LOSC_Event_tutorial'

# name the events and store them in a variable
event_names = np.array(['GW120914', 'LVT151012', 'GW151226', 'GW170104'])

# store the Hanford and livingston files separately (I do it manually to ensure they are in the good order)
fnamesH = np.array(['H-H1_LOSC_4_V2-1126259446-32.hdf5', 'H-H1_LOSC_4_V2-1128678884-32.hdf5', 'H-H1_LOSC_4_V2-1135136334-32.hdf5', 'H-H1_LOSC_4_V1-1167559920-32.hdf5'])
fnamesL = np.array(['L-L1_LOSC_4_V2-1126259446-32.hdf5', 'L-L1_LOSC_4_V2-1128678884-32.hdf5', 'L-L1_LOSC_4_V2-1135136334-32.hdf5', 'L-L1_LOSC_4_V1-1167559920-32.hdf5'])

#initialize strains
strainsH = np.zeros([len(fnamesH), 131072])
strainsL = np.zeros([len(fnamesL), 131072])
#initialize utc
utcH=[]
utcL=[]
#store the data in strainsH and strainsL and utcH, utcL 
for i in range(len(fnamesH)):
    print('Read files', fnamesH[i],' and ',fnamesL[i])
    strainsH[i,:], dt, utc_read1 = read_file(loc+'/'+fnamesH[i])
    strainsL[i,:], dt, utc_read2 = read_file(loc+'/'+fnamesL[i])
    utcH.append(utc_read1)
    utcL.append(utc_read2)


# read the templates in the same order as for the strains
template_names=np.array(['GW150914_4_template.hdf5', 'LVT151012_4_template.hdf5', 'GW151226_4_template.hdf5', 'GW170104_4_template.hdf5'])
# store these in th and tl 
th = np.zeros([len(template_names), 131072])
tl = np.zeros([len(template_names), 131072])
for i in range(len(template_names)):
    th[i,:], tl[i,:] = read_template(loc+'/'+template_names[i])

print('Reading done.')

#---------------------------------------------------------------------------------------------------
# let's answer to a)
print('a) model noise')

#we will use our noise model in noise.py
#we must define a window first to account for boundaries
# we use a cosine window just as in class
x = np.arange(len(strainsH[0]))
x = x - 1.0*x.mean()
window = 0.5*(1 + np.cos(x*np.pi/np.max(x)))

# we now plot the noise models for each event
fig, ax = plt.subplots(4, 2,sharex=True, figsize=(12,10))
fig.suptitle('Noise power spectra for Hanford and Livingston')
for i in range(len(event_names)):
    # the models are computed in plot_noise()
    plot_noise(event_names[i], strainsH[i], strainsL[i], th[i], tl[i], window, i, fig, ax)
plt.savefig('Noise_models.pdf')
plt.close()



f.write('------------------ a) -----------------------------------------------------------------------\n')
f.write('For the noise model, we assume stationnary Gaussian noise. This means we assume \n')
f.write('the noise is not a function of time. We pass a window to the strains using a cosine \n')
f.write('just as we did in class. We also apply a gaussian filer for smoothing the data once \n')
f.write('we have the power spectrum in the real fourier space. The plots are in Noise_models.pdf.')



# -------------------------------- b) --------------------------------
print('b) matched filter')
# we use a matched filter to find the events
# the functions that do that are in noise.py and wave_finder.py

# we need the time for when plotting these events
time=np.linspace(0,len(strainsH[0])/4096,len(strainsH[0]))

# we plot our results
fig, ax = plt.subplots(4, 2,sharex=True, figsize=(12,10))
fig.suptitle('Matched filters for Hanford and Livingston')
for i in range(len(event_names)):
    # the matched filters are computed in this following function in wave_finder.py
    plot_mf(event_names[i], strainsH[i], strainsL[i], th[i], tl[i], window, time, i, fig, ax)
plt.savefig('matched_filters.pdf')
plt.close()


f.write('\n\n------------------ b) --------------------------------------------------------------------\n')
f.write('In this part, we apply a matched filter which is done in wave_finder.py \n')
f.write('and which is applied on the whitened data and templates. The function whitening \n')
f.write('the datasets is defined in noise.py. The plots are shown in matched_filters.pdf.')

#------------------------------------------------------------------------------------
print('c) SNR ')
# here we will compute the signal-to-noise ratios (SNRs) for each
# Hanford, Livingston and Han+Liv

# we will plot all the results together
fig, ax = plt.subplots(4, 3,sharex=True, figsize=(12,10))
fig.suptitle('SNRs for Han, Liv and H+L')
for i in range(len(event_names)):
    # the SNRs are computed in plot_SNR found in wave_finder.py
    plot_SNR(event_names[i], strainsH[i], strainsL[i], th[i], tl[i], window, time, i, fig, ax)
plt.savefig('SNRs.pdf')
plt.close()

f.write('\n\n----------------------c)-------------------------------------------------------------\n')
f.write('In this part we compute the signal-to-noise ratio for each detector \n')
f.write('and for the combined signal. The SNRs are computed in wave_finder.py/SNR()\n')
f.write('and the plotting function is in the same file. The plots are in SNRs.pdf.')


#----------------------------------------------------------------------
print('d) theoretical SNR')

# we will compute the theoretical SNRs from the noise model we have
# and compare this to what we got for the SNRs

# we will plot these all together again
fig, ax = plt.subplots(4, 3,sharex=True, figsize=(13,10))
fig.suptitle('Analytic SNRs for Han, Liv and H+L')
for i in range(len(event_names)):
    # these SNRs are computed in plot_Theo_SNR in wave_finder.py
    plot_Theo_SNR(event_names[i], strainsH[i], strainsL[i], th[i], tl[i], window, time, i, fig, ax)
plt.savefig('Analytic_SNRs.pdf')
plt.close()


f.write('\n\n----------------------d)-----------------------------------------------------------------------\n')
f.write('In this part we compute the ANALYTIC signal-to-noise ratio for each detector \n')
f.write('and both combined. We do this by getting this SNR out of our noise model. We  \n')
f.write('store the plots in Analytic_SNRS.pdf. We can see that the peaks are at the same\n')
f.write('times. We see the Analytic SNRs are usually higher than the SNRs previously calculated \n')
f.write('This can be due to the fact that in some chunks in the data, maybe the noise was not \n')
f.write('exactly stationnary.')

#-----------------------------------------------------------------------
print('e) frequency')

# Here we find the frequency at which half the weight comes from before
# and half from after. We do this for both events

#let's get a frequency array (it's the same for each strain)
# it's just the x array we have been plotting when in F space
frequency = np.fft.rfftfreq(len(strainsH[0]), dt)

f.write('\n\n--------------------------e)--------------------------------------------------------\n')
f.write('In this part, for each event, we compute the frequency at which the weigth \n')
f.write('comes equally from before than from after. Here are the results : \n\n')

# the function computing this mid frequency is mid_freq()
# let's compute it for each event and write the results in our results file
for i in range(len(event_names)):
    f.write('Event : '+event_names[i]+'\n')
    #compute the frequencies
    midh, midl = mid_freq(event_names[i], strainsH[i], strainsL[i], th[i], tl[i], window, frequency)
    f.write('For Hanford, the mid frequency is '+str(midh)+' Hz\n')
    f.write('For Livingston, the mid frequency is '+str(midl)+' Hz\n\n')

#-----------------------------------------------------------------------------------------
print('f) time of arrival')
# here we estimate our precision on our estimate of the time of arrival of the waves
# and our estimate for the uncertainty on their positions
f.write('\n---------------------------f)-------------------------------------------------------\n')

# let's fit a Gaussian to the peak in the signal to noise ratio and this will give us
# a estimate for the arrival time (the mean) and for the error (std dev)
# define our Gaussian
def gauss(x, u, A, sig):
    return A*np.exp(-(x-u)**2/(2.0*sig**2))

f.write('\nHere, we will compute the estimated time of arrival at each detector\n')
f.write('and assuming they are 3000 Km apart, we will estimate the uncertainty \n')
f.write('in the position using the difference in the arrival time for both detectors \n')
f.write('since we know grav. waves travel at speed c. Here are the results : \n')
# perform the fit for each event
for i in range(len(event_names)):
    f.write('\nEvent : '+event_names[i]+'\n')
    # fit the gaussian, the function is found in wave_finder.py
    fith, fitl = time_estimate(event_names[i], strainsH[i], strainsL[i], th[i], tl[i], window, time, gauss)
    # we take sigma in our fit as the error in our estimate
    f.write('The waves arrive at Hanford '+str(fith[0])+' +/- '+str(fith[2])+' seconds\n')
    f.write('after the beginning of the event\n')
    f.write('The waves arrive at Livingston '+str(fitl[0])+' +/- '+str(fitl[2])+' seconds\n')
    f.write('after the beginning of the event.\n\n')
    # we also estimate the positional uncertainty given the difference in the time of arrival
    # and knowing the distance between the detectors are a few thousand Km apart
    deltaT = np.abs(fith[0] - fitl[0])
    pos_inc = (3e8 * deltaT)/3e6
    f.write('Hence, for this event, the positional uncertainty is of about '+str(pos_inc)+'\n\n')



f.close()
print('Done')

