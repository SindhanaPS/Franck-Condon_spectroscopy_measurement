import numpy as np
from func_single import *
import math as m
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression

flag1 = 0
Omega = 20.0
#####################################################
#                Importing data                     #
#####################################################

if flag1 == 0:
   data0 = np.loadtxt('spectrum_N0.txt')
   data1 = np.loadtxt('spectrum_N1.txt')
   data2 = np.loadtxt('spectrum_N10.txt')

   datadiv1 = np.loadtxt('spectrum_div_N1.txt')
   datadiv2 = np.loadtxt('spectrum_div_N10.txt')

   wseries = data0[:,0]
   Anoisy0 = data0[:,1]
   Anoisy1 = data1[:,1]
   Anoisy2 = data2[:,1]

   Pdiv1 = datadiv1[:,1]
   Pdiv2 = datadiv2[:,1]

####################################################

######################################################
#                 Formatting                         #
######################################################

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

rcParams['text.latex.preamble'] = [
       r'\usepackage{physics}',
       r'\usepackage{amsmath}',
]

rcParams['axes.linewidth'] = 1


plt.rc('text', usetex=True)
######################################################


if flag1 == 0:
   for i in range(wseries.size):
      if wseries[i] == Omega:
         mid = i 
   
   font = {'family' : 'Helvetica',
           'weight' : 'normal',
           'size'   : 50}

   plt.rc('font', **font)
   plt.rc('text', usetex=True)

   # Figure 3_c
   fig1, ax1 = plt.subplots(1) 

   y_low, y_high = ax1.get_ylim()
   ax1.set_xlim(15,30)
   ax1.set_ylim(0,0.6)
   ax1.plot(wseries,Anoisy0,color='orange')
   ax1.set_ylabel(r'$A(\omega)+\eta(\omega)$')   
   ax1.set_xlabel(r'$\omega/\omega_a$')   

   plt.savefig('fig3_c.pdf', bbox_inches='tight')
   plt.show()

   # Figure 3_a
   fig1, ax1 = plt.subplots(1)

   y_low, y_high = ax1.get_ylim()
   ax1.set_xlim(15,30)
   ax1.set_ylim(0,0.6)
   ax1.plot(wseries,Anoisy1,color='orange')
   ax1.set_ylabel(r'$A(\omega)+\eta(\omega)$')   
   ax1.set_xlabel(r'$\omega/\omega_a$')   
   ax1.set_xlim(15,30)

   plt.savefig('fig3_a.pdf', bbox_inches='tight')
   plt.show()

   # Figure 3_b
   fig1, ax1 = plt.subplots(1)

   y_low, y_high = ax1.get_ylim()
   ax1.set_xlim(15,30)
   ax1.set_ylim(0,0.6)
   ax1.plot(wseries,Anoisy2,color='orange')
   ax1.set_ylabel(r'$A(\omega)+\eta(\omega)$')   
   ax1.set_xlabel(r'$\omega/\omega_a$')   
   ax1.set_xlim(15,30)

   plt.savefig('fig3_b.pdf', bbox_inches='tight')
   plt.show()

   font = {'family' : 'Helvetica',
           'weight' : 'normal',
           'size'   : 40}

   plt.rc('font', **font)
   plt.rc('text', usetex=True)

   # Figure 3_d
   fig1, ax1 = plt.subplots(1) 

   y_low, y_high = ax1.get_ylim()
   ax1.set_ylim(0,y_high)
   ax1.plot(wseries,Pdiv1, linewidth=10)
   ax1.set_ylabel(r'$P_{\mathrm{div}}(\omega)$')   
   ax1.set_xlabel(r'$\omega/\omega_a$')   
   ax1.set_xlim(-7,7)

   plt.savefig('fig3_d.pdf', bbox_inches='tight')
   plt.show()

   # Figure 3_e
   fig1, ax1 = plt.subplots(1) 

   y_low, y_high = ax1.get_ylim()
   ax1.set_ylim(0,y_high)
   ax1.plot(wseries,Pdiv2, linewidth=10)
   ax1.set_ylabel(r'$P_{\mathrm{div}}(\omega)$')   
   ax1.set_xlabel(r'$\omega/\omega_a$')   
   ax1.set_xlim(-7,7)

   plt.savefig('fig3_e.pdf', bbox_inches='tight')
   plt.show()
