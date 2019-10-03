# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 2 - Q2

#---------------------------------------------
import numpy as np 
from matplotlib import pyplot as plt

# This problem wants us to use Newton's method in order to fit the flare in the 
# stellar flux data by minimizing Chi squared


# a) Data and Guess--------------------------------------------------------
print('---------------a)----------------------')
# The first part is to isolate the flare in the data and write an 
# exponential decay guess

data=np.loadtxt('229614158_PDCSAP_SC6.txt',delimiter=',')

#get the time and flux
t=data[:,0]
flux=data[:,1]

# We zoom on the flare by cutting the data around the peak
# By investigating the file we see the flare is at the 3200th point
t=t[3200:3275]
flux=flux[3200:3275]

# We will model an exponential decay to this peak
# We first define a function for our exponential decay
def exp_decay(x, params):
	return np.exp(-(x-params[0])/params[1])+params[2]
#so params is an array holding X0, a scaling factor and a shift in the y direction

print('So, we will model an exponential decay, which is clearly non-linear')
print('The function will have the form ')
print('exp_decay=exp(-(x-x0)/f)+c')

print('')
# We construct our guess decay
print('We start guessing/investigating the parameters')
# By looking at the data, we can see the data flattens out at y=1
y_shift=1
print('We see the data flattens out at y=', y_shift, 'which is our guess for c')
print('')
print('We control how fast the decay is with the factor f')
print('0.025 seems to do the trick')
# We find a scale factor 
f=0.025

# we want our X0 to be the x coordinate of the peak
x0_index=np.where(data[:,1]==data[:,1].max())
X0=data[x0_index][0][0]
#we will make this X0 relative to the data (shift and scaling)
flare=data[x0_index][0][1]-y_shift
X0=np.log(flare)*f+X0
print('')
print('We find our guessed X0 by finding the position of the flare and')
print('making it into the exponential scale with both previous parameters')
print('')


#We store these guess params in an array
guess_p=[X0, f, y_shift]
# we compute the guess
guess=exp_decay(t, guess_p)

# We plot this with our initial guess
plt.clf()
plt.plot(t,flux, label='Stellar flux data')
plt.plot(t, guess, label='Initial guess')
plt.xlabel('time')
plt.ylabel('flux')
plt.legend()
plt.savefig('initial_guess_2a.pdf')
plt.close()



#b) Newton's Method---------------------------------------------------------------
print('-------------------b)---------------------')
#Here, we will fit the data with our exponential decay by minimizing Chi squared

#In order to use this method, we first compute all derivatives of our function
def calc_exp_decay(t, params):
	y=exp_decay(t, params) #this is our guess
	grad=np.zeros([t.size,  params.size])
	# we take derivative wrt all the parameter
	exp_term=np.exp(-(t-params[0])/params[1]) #recurrent in the derivatives
	# first wrt params[0], X0
	grad[:,0]=1/params[1]*exp_term
	#then wrt to params[1], f
	grad[:,1]=1/params[1]**2*(t-params[0])*exp_term
	#then wrt to params[2], c
	grad[:,2]=1

	return y, grad



# Then we define our Newton method function as in the example from class
p=guess_p.copy() #the guess parameters
p=np.array(p)
for j in range(5):
	guess,grad=calc_exp_decay(t, p)
	r=flux-guess
	chi2=(r**2).sum()
	r=np.matrix(r).transpose()
	grad=np.matrix(grad)

	lhs=grad.transpose()*grad
	rhs=grad.transpose()*r
	dp=np.linalg.inv(lhs)*rhs
	for jj in range(len(p)):
		p[jj]=p[jj]+dp[jj]

	#stop when the fit is good. 
	if chi2<1 and j>0:
		break

#So now, p are our Newton fit parameters and chi2 is our Chi squared
print('')
print('We do the Newton fit')
print('Our best fit parameters are :')
print('X0 = ', p[0])
print('f = ', p[1])
print('c = ', p[2])
print('Also, Chi squared is now', chi2)
print('')

Newton_model=exp_decay(t, p) # our new fit
initial=exp_decay(t,guess_p) #our initial guess

plt.clf()
plt.plot(t,flux, label='stellar data')
plt.plot(t, initial, label='initial guess')
plt.plot(t, Newton_model, label='Newton method fit', color='r')
plt.xlabel('time')
plt.ylabel('flux')
plt.legend()
plt.savefig('newton_fit.pdf')

# c) Errors -----------------------------------------------------------------
print('-------------------c)---------------------')
# we go on and estimate our errors for our results
# Recall p are our fit parameters and chi2 is our chi-squared
noise=np.diag(1/(flux-guess)**2) #get the noise
# we compute covariance matrix step by step
gtn=np.dot(grad.transpose(), noise)
cov=np.dot(gtn, grad)
cov=np.linalg.inv(cov)

# To get the errors in our parameters, we take the square root 
# of the diagonal of our covariance matrix
errors=np.sqrt(np.diag(cov))

print('')
print('The errors in the parameters are :')
print('Error in x0 =', errors[0])
print('Error in f =', errors[1])
print('Error in c =', errors[2])
print('')

# d) Full Span-----------------------------------------------------
print('------------------d)------------------')
print('')
print('When looking at the full data, we see that there are some regularities')
print('in the data that are not taken into account when isolating the flare')
print('so I think our error is not a good estimate for this data. ')









