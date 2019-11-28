import numpy as np 

# here I will define some functions used many times and
# regarding the numerical algorithm

# we define a vector operation function as in class
def Ax(V, mask):
	# we copy the current V
    Vuse=V.copy()
    # put zero in the mask
    Vuse[mask]=0
    # use the algorithm
    ans=(Vuse[1:-1,:-2]+Vuse[1:-1,2:]+Vuse[2:,1:-1]+Vuse[:-2,1:-1])/4.0
    ans=ans-V[1:-1,1:-1]
    return ans	

# We defin the padding function just as in class
def pad(A):
    AA=np.zeros([A.shape[0]+2,A.shape[1]+2])
    AA[1:-1,1:-1]=A
    return AA