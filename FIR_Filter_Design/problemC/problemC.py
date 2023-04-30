#Python libraries for math and graphics
import numpy as np
from numpy import cos, sin, pi, absolute, arange
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show,savefig
import math
from cvxpy import *

##############################################################
# Given wc (Cut off frequency) and alpha (Attenuation constant), 
# we need to minimize N (Order of the filter) with
# wc Filter coefficients as objective variables. We will start at minimum value of N 
##########################################################
alpha = 0.006

wc = pi/2.5
wp = pi/3


# Create optimization variables
N = 1
#Keep increasing N and check whether the solution exists or not.
while (True) :
    sz = N*15 
    w = np.linspace(0,pi, sz)
    k = np.arange(0,N+1).reshape(N+1,1)
    coskw = cos(k*w).T

    wi = np.where(w<= wp)[0].max()
    wo = np.where(w>= wc)[0].min()

    print("N=", N)
    an = Variable((N+1,1))
    # Form objective function
    obj = Minimize(0)
    
    #Constraints
    constraints = [ coskw[0:wi, :]@an <= 1.12, coskw[0:wi,:]@an >= 0.89, cos(wp*k).T@an >= 0.89, cos(wp*k).T@an <= 1.12,  coskw[wo:sz, :]@an <= alpha , coskw[wo:sz, :]@an >= -alpha, cos(wc*k).T@an <= alpha, cos(wc*k).T@an >= -alpha ]  

    # Form and solve problem.
    prob = Problem(obj, constraints)
    prob.solve()

    print(prob.status)
    if (prob.status == "optimal"):
        break
    N += 1 

taps = an.value[:,0]

#------------------------------------------------
# Plot the magnitude response of the filter.
#------------------------------------------------

figure(1)
clf()
hw = cos(k*w).T@taps  
plot(w, absolute(hw), linewidth=2)
xlabel('w in radians')
ylabel('|H(w)|')
title('Frequency Response (Optimal N=%d Attenuation=%0.3f)'% (N ,alpha))
ylim(-0.1, 1.2)
grid(True)

savefig('./figs/HW_with_Alpha_.006.png')
show()
