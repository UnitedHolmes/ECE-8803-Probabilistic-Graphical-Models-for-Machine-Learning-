# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 23:51:32 2018

@author: UnitedHolmes
"""

import numpy as np
from scipy import optimize
from scipy.linalg import sqrtm
import time
#from sys import exit

data = np.load("house_votes_clean.npy","r")

X_demo_all = data[np.where(data[:,0]>0)]

X_demo = X_demo_all[:,1:]
holdout = int(np.round(0.8*len(X_demo[:,0])))
X = X_demo[0:holdout,:]
X_holdout = X_demo[holdout:,:]

#p = 10 #Number of nodes in the graph
p = len(X[0,:])

#n = 100 #Number of datapoints
n = len(X[:,0])

mu = 0.01

# Regularization Parameter
lamb = np.sqrt(np.log(n)/p)
#lamb = 0.001

np.random.seed(2018)

# Matrix containing signed edge values between all edges : This is a symmetric matrix
# Can potentially be replaced by a dictionary of edge values
theta = np.random.rand(p,p)
np.fill_diagonal(theta,1)

# Sample data of 0-mean gaussians : n samples each length p
#X = np.load('house_votes.npy')
#X = np.random.multivariate_normal([0]*p, np.identity(p), (n))

# Replace positive elements by 1 and negative elements by -1 
#X = np.where(X >= 0,1,-1)


def cond(X,theta,r):
    """ Return the Conditional Probability of x_r given x/r for one sample of data. 
        Arguments: 
        X - One sample of data, a px1 vector
        theta - The Corresponding theta Vector from the full theta matrix
        r = The node whose neighbourhood is to be estimated 
    """
    x_r = X[r]

    x_nr = np.array([X[i] for i in range(len(X)) if i!=r]) # leave out 'r'th column/element when writing to x_nr

    temp = np.exp(2 * x_r * (theta.dot(x_nr)))      #Evaluating the numerator in the conditional probability
    return(temp/(temp+1))                           #Return Conditional probability of x_r given x_nr

def reg_func(X_nr,theta_nr, param=None):
    """ Return the Normalized Regularization term
        Arguments:
        X_nr     : One sample of data with the 'r'th column removed
        theta_nr : The corresponding theta vector with the rth element removed
        param    : If param=1 do normalized l1 regularization else do trace lasso
    """
    if(param==1):
        l1_norm = 0
        for i in range(len(theta_nr)):
            l1_norm += np.abs(theta_nr[i]) * np.linalg.norm(X_nr[:,i])  #Return normalized l1 norm
        return l1_norm
    elif (param==2):
        en_norm = 0
        for i in range(len(theta_nr)):
            en_norm += 0.5*np.abs(theta_nr[i]) * np.linalg.norm(X_nr[:,i]) + 0.5*np.abs(theta_nr[i])**2 * np.linalg.norm(X_nr[:,i])**2  #Return elastic net norm
        return en_norm
    else:
        global mu
        if(mu > 1e-5):
           mu /= 10

        # M is the matrix that we want to compute the trace Lasso for - so X * Diag(theta)
        M = X_nr.dot(np.diag(theta_nr))
        S = sqrtm(M.dot(np.transpose(M)) + mu * np.identity(n))
        Sinv = np.linalg.inv(S)
        return 0.5 * theta_nr.dot(np.diag(np.diagonal(np.transpose(X_nr).dot(Sinv).dot(X_nr)))).dot(theta_nr)

def obj_func(theta,r):
    """ The function to compute the Pseudo-Likelihood, the Regularization term
    """

    objVal = 0

    
    # Compute Pseudo-Likelihood for each data sample (Each row of X) and sum them
    
    for i in range(n):
        objVal -= np.log(cond(X[i,:],theta,r))
    objVal /= n
    # Delete 'r'th column from the matrix
    X_nr = np.delete(X,r,1)

    #### Uncomment first line and comment second line to get Lasso regularization
#    objVal += lamb*reg_func(X_nr,theta,1) #lasso  
#    objVal += lamb*reg_func(X_nr,theta,2) #Elastic Net  
    objVal += lamb*reg_func(X_nr,theta)  #Trace lasso

    return objVal

"""Perform Signed Edged recovery for every edge in the graph"""
start_time = time.time()
for i in range(p):
    theta_arg = []
    """ The following loop constructs the theta vector for this particular node
        and basically eliminates duplicates.
        The idea is that we want only the right upper triangular portion of theta matrix
        to be changed i.e the edge between 1 and 0 is theta[0,1] and not theta[1,0]
        Similarly edge between 4 and 2 is theta[2,4] and not theta[4,2]
        So if i=2, theta_arg contains ([0,2],[1,2],[2,3],[2,4],...,[2,9])
    """
    for j in range(p):
        if(i != j):
            theta_arg.append(theta[i,j])

    theta_arg = np.array(theta_arg) 



    #Run the Optimizer on the Objective function, method is Conjugated Gradients
    theta_opt = optimize.minimize(obj_func, theta_arg,args=(i,),method='CG', options={'disp':True})


    
    """
    # Write the elements of minimized theta to their right place in the overall theta matrix
    # Here we are writing from a length p-1 vector to a pxp Matrix and basically 
    # performing the reverse of the loop in the beginning of the function. 
    """
    for j in range(p-1):
        if(i <= j):
            theta[i,j+1] = theta_opt.x[j]
        if(i > j):
            theta[i,j] = theta_opt.x[j]

time_elapsed = time.time() - start_time

print("Time Elapsed: ", time_elapsed)

print(theta)

for i in range(p):
    for j in range(i):
        if(np.abs(theta[i,j]) < np.abs(theta[j,i])):
            theta[i,j],theta[j,i] = np.sign(theta[i,j]), np.sign(theta[i,j])
        else:
            theta[i,j],theta[j,i] = np.sign(theta[j,i]), np.sign(theta[j,i])

#exit()

np.save("trial3_results",theta)
np.savetxt("trial3_results.txt", theta, fmt='%.0d', delimiter=', ', newline=', \n ', header='Reg = Trace Lasso, theta = \n')
#with open("trial3_results.txt", "a") as text_file:

#    print("Reg = Trace Lasso, theta = \n %s \n"%(theta),file=text_file)

### calculate holdout likelihood
#log_likelihood = 0
accuracy = np.zeros([1,p])
for i in range(0,len(X_holdout[:,0])):
#    this_likelihood = 0
    for j in range(0,p):
        this_theta = []
        for r in range(0,p):
            if (j != r):
                this_theta.append(theta[j,r])
        
        this_theta = np.array(this_theta)
        this_cond = cond(X_holdout[i,:],this_theta,r)
        if (this_cond*2 - 1)/X_holdout[i,j] > 0:
            accuracy[0,j] += 1
            
accuracy_per = accuracy / len(X_holdout[:,0])
    