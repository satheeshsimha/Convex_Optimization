#Python libraries for math and graphics
import numpy as np
from numpy import cos, sin, pi, absolute, arange
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show,savefig
import math
from cvxpy import *

##########################################################
# Given N (Order of the filter) and wc (Cut off frequency),
# we need to minimize alpha (Attenuation constant) with
# alpha and Filter coefficients as objective varaibles 
##########################################################

N= 20 
sz = 15*N 
w = np.linspace(0,pi, sz)
wc = pi/2.5
wp = pi/3
k = np.arange(0,N+1).reshape(N+1,1)
coskw = cos(k*w).T

wi = np.where(w<= wp)[0].max()
wo = np.where(w>= wc)[0].min()
    
# Create optimization variables
an = Variable((N+1,1))
alpha = Variable()

# Form objective function
obj = Minimize(alpha)

#Constraints
constraints = [ coskw[0:wi, :]@an <= 1.12, coskw[0:wi,:]@an >= 0.89, cos(wp*k).T@an >= 0.89, cos(wp*k).T@an <= 1.12,  coskw[wo:sz, :]@an <= alpha , coskw[wo:sz, :]@an >= -alpha, cos(wc*k).T@an <= alpha, cos(wc*k).T@an >= -alpha ]  

# Form and solve problem.
prob = Problem(obj, constraints)
prob.solve()

print(prob.status)

print("alpha=", alpha.value)
print("an =", an.value)

taps = an.value[:,0]
print(taps)

#------------------------------------------------
# Plot the magnitude response of the filter.
#------------------------------------------------

figure(1)
clf()
hw = cos(k*w).T@taps  
plot(w, absolute(hw), linewidth=2)
xlabel('Frequency (Hz)')
ylabel('Gain')
title('Frequency Response (%d Taps)'% N)
ylim(-0.1, 1.2)
grid(True)

savefig('./figs/HW_with_N20.png')
show()
