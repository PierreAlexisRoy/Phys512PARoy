# Pierre-Alexis Roy 
# 260775494
# Phys 512 Problem Set 3 - Q2

#---------------------------------------------
import numpy as np
import camb
from matplotlib import pyplot as plt
from wmap_camb_example import *

# In this problem we will write and use a Newton's method/
# Lavenberg-Marquardt to fit the camb model to the data

# We first do this while keeping tau = 0.05
def get_spectrum_tauFixed(pars,lmax=1500):
	#print('pars are ',pars)
	H0=pars[0]
	ombh2=pars[1]
	omch2=pars[2]
	tau=0.05
	As=pars[3]
	ns=pars[4]
	pars=camb.CAMBparams()
	pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
	pars.InitPower.set_params(As=As,ns=ns,r=0)
	pars.set_for_lmax(lmax,lens_potential_accuracy=0)
	results=camb.get_results(pars)
	powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
	cmb=powers['total']
	tt=cmb[:,0]
	return tt


# For the newton method we need the gradient, we define a function
# that returns the double sided derivative
def get_guess_grad(p, data, func):
	#compute our guess
	#y = func(p)[2:len(data)+2]
	# create an empty array in which we will store the grad
	grad=np.zeros([len(data),len(p)])
	# compute the grad for each parameter via 1st order numerical derivative
	for i in range(len(p)):
		#create a copy of the parameters so they start from p each time
		newp = p.copy() 
		# compute a delta for numerical derivative
		delta = 0.001*p[i] 

		# Compute the first term in the derivative
		newpUp = newp[i] + delta 
		newp[i] = newpUp # update the differentiated parameter
		funcUp = func(newp)[2:len(data)+2]

		# Compute the second term
		newpDown = newp[i] - 2*delta
		newp[i] = newpDown #update the parameter for second term
		funcDown = func(newp)[2:len(data)+2]

		grad[:,i] = (funcUp - funcDown)/(2*delta)
	return grad


# we write out our Newton-Lavender-Marquadt minimizer
def Newton_LM(p, data, func, noise):
	old_chi2 = 2000
	lam = 0.0001
	grad = get_guess_grad(p, data, func)
	pnew = p.copy()
	for j in range(15):
		print('Newton_LM loop', j)
		#guess, grad = get_guess_grad(p, data, func)
		guess = func(pnew)[2:len(data)+2]
		r=data-guess
		chi2=(r**2/noise**2).sum()
		print('chi2 is', chi2)
		#print('improvement is', (old_chi2-chi2))
		# From here, we check if we are satsfied with our result
		# and one of 3 options happen
		if np.abs(chi2-old_chi2) < 0.01 and j>0 and chi2 <= old_chi2:
			# this means we are converged !
			print('We are converged and Chi squared is', chi2)
			p = pnew.copy()
			# we go on and compute the errors
			# we start by covariance matrix
			error = np.diag(1.0/noise**2)
			cov = np.dot(grad.transpose(), error)
			cov = np.dot(cov, grad)
			cov = np.linalg.inv(cov)
			# error in the parameters
			fitp_error = np.sqrt(np.diag(cov))
			break

		elif old_chi2 < chi2:
			# we got worst on last step, increase lam for Lavenberg-M.
			print('Chi2 increased, increase the lambda in L-M')
			lam = 200*lam
			pnew = p.copy()


		else:
			# chi2 got lower but it is not yet converged
			#update chi2
			old_chi2 = chi2
			p = pnew.copy()
			#reduce lambda
			lam = lam/100.0
			# get gradient
			grad = get_guess_grad(p, data, func)

		# Newton-LM algorithm
		r=np.matrix(r).transpose()
		grad=np.matrix(grad)
		lhs=grad.transpose()*np.diag(1.0/noise**2)*grad
		lhs+=lam*np.diag(np.diag(lhs))
		rhs=grad.transpose()*np.diag(1.0/noise**2)*r
		dp=np.linalg.inv(lhs)*rhs
		# update the parameters
		for jj in range(len(p)):
			pnew[jj]=pnew[jj]+dp[jj]
		print('We keep chi2 = ', chi2)
		print('')
	return p, cov, fitp_error



# # test get_guess_grad
# pars = np.asarray([65,0.02,0.1,2e-9,0.96])
# data = wmap[:,1]
# guess,grad = get_guess_grad(pars, data, get_spectrum_tauFixed)


# We perform the fit
print('Fit the data for Tau fixed')
print('')
pars = np.asarray([65,0.02,0.1,2e-9,0.96])
data = wmap[:,1]
noise=wmap[:,2]
fitp, fitpcov, fitperr = Newton_LM(pars, data, get_spectrum_tauFixed, noise)

# We print the results
print('')
print('The fit parameters are :')
print('')
print('H0                  =', fitp[0],' +/- ', fitperr[0])
print('Density baryonic    =', fitp[1],' +/- ', fitperr[1])
print('Density dark matter =', fitp[2],' +/- ', fitperr[2])
print('Amplitude           =', fitp[3],' +/- ', fitperr[3])
print('Tilt                =', fitp[4],' +/- ', fitperr[4])


# Evaluating our derivative estimate
print('')
print('----------- Do we trust our gradient ? -------------------')
print('')
print('We see that the Chi square is decreasing in Newton algorithm')
print('This is a sign that the gradient estimate we do is good, otherwise')
print('we would not see a persistent chi square decrease.')
print('')

print('----------- If we were to float Tau ---------------------')

# we will compute the covariance matrix with tau = 0.05
fitp_full = np.insert(fitp, 3, 0.05)
grad_full = get_guess_grad(fitp_full, data, get_spectrum)
error_full = np.diag(1.0/noise**2)
cov_full = np.dot(grad_full.transpose(), error_full)
cov_full = np.dot(cov_full, grad_full)
cov_full = np.linalg.inv(cov_full)
error_full = np.sqrt(np.diag(cov_full))

print('We see the error on tau in the full covariance matrix is huge compared to tau')
print('tau =', 0.05,' +/- ', error_full[3])
print('So we must have that Tau is not really correlated to the other parameters')
print('Hence, floating tau would not affect our errors much. ')





# # Compute best model
# print('')
# print('computing best fit')
# best_model=get_spectrum_tauFixed(fitp)[2:len(data)+2]

# # plot the result
# plt.clf()
# plt.plot(wmap[:,0], data, '.')
# plt.plot(best_model)
# plt.show()

# p=guess_p.copy() #the guess parameters
# p=np.array(p)
# for j in range(5):
# 	guess,grad=calc_exp_decay(t, p)
# 	r=flux-guess
# 	chi2=(r**2).sum()
# 	r=np.matrix(r).transpose()
# 	grad=np.matrix(grad)

# 	lhs=grad.transpose()*grad
# 	rhs=grad.transpose()*r
# 	dp=np.linalg.inv(lhs)*rhs
# 	for jj in range(len(p)):
# 		p[jj]=p[jj]+dp[jj]

# 	#stop when the fit is good. 
# 	if chi2<1 and j>0:
# 		break

#first version

# # we write out our Newton-Lavender-Marquadt minimizer
# def Newton(p, data, func, noise):
# 	old_chi2 = 1600
# 	lam = 0
# 	for j in range(15):
# 		#print('looping Newton', j)
# 		#print('running get_guess_grad')
# 		guess, grad = get_guess_grad(p, data, func)
# 		r=data-guess
# 		chi2=(r**2/noise**2).sum()
# 		#print('chi2 is', chi2)
# 		#print('improvement is', (old_chi2-chi2))
# 		if old_chi2 < chi2:
# 			if lam < 1e12:
# 				lam = 1000*(lam+5)
# 		else:
# 			lam /= 1000
# 		r=np.matrix(r).transpose()
# 		grad=np.matrix(grad)

# 		lhs=grad.transpose()*grad
# 		lhs+=lam*np.diagonal(lhs)
# 		rhs=grad.transpose()*r
# 		dp=np.linalg.inv(lhs)*rhs
# 		for jj in range(len(p)):
# 			p[jj]=p[jj]+dp[jj]

# 		print('loop', j, ', old_chi = ', old_chi2,', new_chi2 = ', chi2 )
# 		#stop when the fit is good. 
# 		if np.abs(chi2-old_chi2) < 0.1 and j>0 and chi2 < old_chi2:
# 			break
# 		elif old_chi2 >= chi2:
# 			old_chi2 = chi2
# 			print(chi2)

# 		else:
# 			print('Chi2 grew, Use Lavenberg-Marquardt')
# 	return p