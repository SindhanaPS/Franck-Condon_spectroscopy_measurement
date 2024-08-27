import numpy as np
from func_single import *
import math as m
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression

flag1 = 4
flag2 = 2
Omega = 20.0

###############################################################
##### flag1 = 0                      plot Figures 1d-f
##### flag1 = 1  & flag2 = 0,1,2     process data for Figure 2
##### flag1 = 2                      plot Figure 2
##### flag1 = 3                      plot Figure S1
##### flag1 = 4                      plot Figure S2
##### flag1 = 5                      plot Figure S3
###############################################################

#####################################################
#                Importing data                     #
#####################################################

if flag1 == 0:
   data0 = np.loadtxt('spectrum_N0_single_large.txt')
   data1 = np.loadtxt('spectrum_N0_single.txt')
   data2 = np.loadtxt('spectrum_N8_single.txt')

   wseries = data0[:,0]
   Anoisy01 = data0[:,1]
   Anoisy0 = data1[:,1]
   Anoisy8 = data2[:,1]

if flag1 == 1:
   data = np.loadtxt('Dd.txt')
   Dd = data
   data = np.loadtxt(f'Dd_0.txt')
   N1 = data[:,0]
   N = np.zeros([Dd.size,N1.size])
   meanDd = np.zeros([Dd.size,N1.size])
   stdDd = np.zeros([Dd.size,N1.size])
   diffDd = np.zeros([Dd.size,N1.size])

   for i in range(Dd.size):
      data = np.loadtxt(f'Dd_{i}.txt')
      N[i] = data[:,0]
      meanDd[i] = data[:,1]
      stdDd[i] = data[:,2]
      diffDd[i] = (meanDd[i]-Dd[i])*100/Dd[i]

if flag1 == 2:
   data1 = np.loadtxt('DdN_eta_1.txt')
   data2 = np.loadtxt('DdN_eta_4.txt')
   data3 = np.loadtxt('DdN_eta_9.txt')

   N = data1[:,0]
   Ddmin1 = data1[:,1]
   Ddmin2 = data2[:,1]
   Ddmin3 = data3[:,1]

   eta1 = 0.1 
   eta2 = 0.1/4
   eta3 = 0.1/9

if flag1 == 3:
   data = np.loadtxt('EstimateDd.txt')

   Dd = data[:,0]
   meanDd0 = data[:,1]
   stdDd0 = data[:,2]
   meanDdtherm = data[:,3]
   stdDdtherm = data[:,4]
   meanDdfock = data[:,5]
   stdDdfock = data[:,6]
   meanDdcoh = data[:,7]
   stdDdcoh = data[:,8]

if flag1 == 4:
   dataCaH = np.loadtxt('CaH_M2-M12.txt')
   dataCaHapprox = np.loadtxt('CaH_M2-M12_approx.txt')
   dataSrF = np.loadtxt('SrF_M2-M12.txt')
   dataSrFapprox = np.loadtxt('SrF_M2-M12_approx.txt')

   nCaH = dataCaH[:,0]
   M2M1CaH = dataCaH[:,1]
   nCaHline = dataCaHapprox[:,0]
   gnCaH = dataCaHapprox[:,1]
   nSrF = dataSrF[:,0]
   M2M1SrF = dataSrF[:,1]
   nSrFline = dataSrFapprox[:,0]
   gnSrF = dataSrFapprox[:,1]

if flag1 == 5:

   dataCaH = np.loadtxt('CaH.txt')
   dataSrF = np.loadtxt('SrF.txt')

   nCaH = dataCaH[:,0]
   nSrF = dataSrF[:,0]

   percCaH = dataCaH[:,3]
   percSrF = dataSrF[:,3]

####################################################

######################################################
#                 Formatting                         #
######################################################
font = {'family': 'Helvetica',
        'weight': 'normal',
        'size': 18}
rc('font', **font)

# Enable LaTeX and set up the preamble
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{physics} \usepackage{amsmath}'

# Set the linewidth for axes
rcParams['axes.linewidth'] = 1

# Alternatively, you can use plt.rc for consistency
plt.rc('text', usetex=True)
######################################################

if flag1 == 0:
   for i in range(wseries.size):
      if wseries[i] == Omega:
         mid = i 

   # Figure 1_e
   fig1, ax1 = plt.subplots(1) 

   ax1.bar(wseries,Anoisy0, color='orange', alpha=0.3)
   ax1.bar(wseries[mid:mid+2],Anoisy0[mid:mid+2], color='orange')
   ax1.set_ylabel(r'$A(\omega)+\eta(\omega)$', fontsize=20)   
   ax1.set_xlabel(r'$\omega/\omega_a$', fontsize=20)   

   ratio = 0.5
   x_leftb, x_rightb = ax1.get_xlim()
   y_lowb, y_highb = ax1.get_ylim()
   ax1.set_aspect(abs((x_rightb-x_leftb)/(y_lowb-y_highb))*ratio)

   plt.savefig('fig1_e.pdf', bbox_inches='tight')
   plt.show()

   # Figure 1_d
   fig1, ax1 = plt.subplots(1)

   ax1.bar(wseries,Anoisy01, color='orange', alpha=0.3)
   ax1.bar(wseries[mid:mid+3],Anoisy01[mid:mid+3], color='orange')
   ax1.set_ylabel(r'$A(\omega)+\eta(\omega)$', fontsize=20)   
   ax1.set_xlabel(r'$\omega/\omega_a$', fontsize=20)   

   ratio = 0.5
   x_left, x_right = ax1.get_xlim()
   y_low, y_high = ax1.get_ylim()
   ax1.set_ylim(0,y_highb)
   ax1.set_aspect(abs((x_right-x_left)/(y_low-y_highb))*ratio)

   plt.savefig('fig1_d.pdf', bbox_inches='tight')
   plt.show()

   # Figure 1_f
   fig1, ax1 = plt.subplots(1)

   ax1.bar(wseries,Anoisy8, color='orange', alpha=0.3)
   ax1.bar(wseries[mid-1:mid+2],Anoisy8[mid-1:mid+2], color='orange')
   ax1.set_ylabel(r'$A(\omega)+\eta(\omega)$', fontsize=20)   
   ax1.set_xlabel(r'$\omega/\omega_a$', fontsize=20)   

   ratio = 0.5
   x_left, x_right = ax1.get_xlim()
   y_low, y_high = ax1.get_ylim()
   ax1.set_ylim(0,y_highb)
   ax1.set_aspect(abs((x_right-x_left)/(y_low-y_highb))*ratio)

   plt.savefig('fig1_f.pdf', bbox_inches='tight')
   plt.show()

if flag1 == 1:

   if flag2 == 0:
      fname = 'DdN_eta_1.txt'
   elif flag2 == 1:
      fname = 'DdN_eta_4.txt'
   elif flag2 == 2:
      fname = 'DdN_eta_9.txt'

   minDd = np.zeros(N1.size)

   for j in range(N1.size):
      idx = np.argwhere(np.diff(np.sign(diffDd[:,j] - 10))).flatten()
      print(N1[j],min(Dd[idx]))
      minDd[j] = min(Dd[idx])

   lN = 4*np.divide(np.sqrt(N1+0.5),N1+1)

   model = LinearRegression().fit(lN.reshape((-1, 1)), minDd)

   rsq = model.score(lN.reshape((-1, 1)), minDd)
   print(f"coefficient of determination: {rsq}")
   print(f"intercept: {model.intercept_}")
   print(f"slope: {model.coef_}")

   np.savetxt(fname,np.transpose([N1,minDd]))

if flag1 == 2:
   fig1, ax1 = plt.subplots(1) 
  
   lN = 4*np.divide(np.sqrt(N+0.5),N+1)
   
   ax1.scatter(lN, Ddmin1, color = '#a70000')
   ax1.scatter(lN, Ddmin2, color = '#ff5252')
   ax1.scatter(lN, Ddmin3, color = '#ffbaba')

   model1 = LinearRegression().fit(lN.reshape((-1, 1)), Ddmin1)
   model2 = LinearRegression().fit(lN.reshape((-1, 1)), Ddmin2)
   model3 = LinearRegression().fit(lN.reshape((-1, 1)), Ddmin3)

   m1 = model1.coef_
   c1 = model1.intercept_ 
   m2 = model2.coef_
   c2 = model2.intercept_ 
   m3 = model3.coef_
   c3 = model3.intercept_ 

   x = np.linspace(min(lN),max(lN),100)

   ax1.plot(x, m1*x+c1, color = '#a70000', label = r'$\eta_0=%.3f$' %eta1)
   ax1.plot(x, m2*x+c2, color = '#ff5252', label = r'$\eta_0=%.3f$' %eta2)
   ax1.plot(x, m3*x+c3, color = '#ffbaba', label = r'$\eta_0=%.3f$' %eta3)

   labels = [r'$l_{4}$',r'$l_{8}$',r'$l_{16}$',r'$l_{32}$',r'$l_{64}$',r'$l_{128}$',r'$l_{256}$']
   positions = lN[0:-2]
   plt.xticks(positions, labels)
   ax1.set_xlabel(r'$l_n$', fontsize=20)   
   ax1.set_ylabel(r'$\Delta d^*_{\mathrm{lim}}$', fontsize=20)
   ax1.legend(loc='upper left',fontsize=14)
   plt.savefig('fig2.pdf', bbox_inches='tight')
   ax1.set_xlim(left=0)
   ax1.set_ylim(bottom=0)
   plt.show()  

if flag1 == 3:
   fig1, ax1 = plt.subplots(1)

   ax1.plot(Dd,Dd, linewidth=2, label='exact', color='black', linestyle='--')
   ax1.plot(Dd, meanDdtherm, linewidth=2, label='Thermal', color='limegreen', linestyle='-')
   ax1.fill_between(Dd,meanDdtherm+stdDdtherm,meanDdtherm-stdDdtherm, color='limegreen', alpha=0.2)
   ax1.plot(Dd, meanDdcoh, linewidth=2, label='Coherent', color='blue', linestyle='-')
   ax1.fill_between(Dd,meanDdcoh + stdDdcoh, meanDdcoh-stdDdcoh, color='blue', alpha=0.2)
   ax1.plot(Dd, meanDdfock, linewidth=2, label='Fock', color='red', linestyle='-')
   ax1.fill_between(Dd, meanDdfock + stdDdfock, meanDdfock - stdDdfock, color='red', alpha=0.2)
   ax1.set_ylabel(r'$\Delta d_{\mathrm{measured}}^*$', fontsize=20)
   ax1.set_xlabel(r'$\Delta d^*$', fontsize=20)
   ax1.legend(loc='upper left',fontsize=14)
   ax1.set_xlim(left=0)
   ax1.set_ylim(bottom=0)
   fig1.savefig('figS1.pdf',bbox_inches='tight')

   plt.show()

elif flag1 == 4:
   fig1, ax1 = plt.subplots(1)

   ax1.scatter(nCaH,M2M1CaH,label='exact')
   ax1.plot(nCaHline,gnCaH,label=r'$(\Delta d^*)^2g(n)$')
   ax1.set_ylabel(r'$(M_2-M_1^2)/\omega_a^2$', fontsize=20)
   ax1.set_xlabel(r'$n$', fontsize=20)
   ax1.legend(loc='upper left', fontsize=16)
   ax1.set_title("CaH", fontsize=20)
   fig1.savefig('figS2a.pdf',bbox_inches='tight')
   plt.show()

   fig1, ax1 = plt.subplots(1)

   ax1.scatter(nSrF,M2M1SrF,label='exact')
   ax1.plot(nSrFline,gnSrF,label=r'$(\Delta d^*)^2g(n)$')
   ax1.set_ylabel(r'$(M_2-M_1^2)/\omega_a^2$', fontsize=20)
   ax1.set_xlabel(r'$n$', fontsize=20)
   ax1.set_title("SrF", fontsize=20)
   ax1.legend(loc='upper left', fontsize=16)
   fig1.savefig('figS2b.pdf',bbox_inches='tight')
   plt.show()

elif flag1 == 5:
   fig1, ax1 = plt.subplots(1)

   ax1.plot(nCaH,percCaH, color='lightgrey', linestyle='dashed',zorder=1)
   ax1.scatter(nCaH,percCaH, label='CaH',color='tab:red',zorder=2)
   ax1.set_ylabel(r'Error $|\frac{\Delta d_{\mathrm{measured}}-\Delta d}{\Delta d}|\times 100\%$', fontsize=18)
   ax1.set_xlabel(r'$n$', fontsize=18)
   ax1.legend(loc='upper right', fontsize=16)
   fig1.savefig('figS3a.pdf',bbox_inches='tight')
   plt.show()

   fig1, ax1 = plt.subplots(1)

   ax1.plot(nSrF,percSrF, color='lightgrey', linestyle='dashed',zorder=1)
   ax1.scatter(nSrF,percSrF, label='SrF', color='tab:red',zorder=2)
   ax1.set_ylabel(r'Error $|\frac{\Delta d_{\mathrm{measured}}-\Delta d}{\Delta d}|\times 100\%$', fontsize=18)
   ax1.set_xlabel(r'$n$', fontsize=18)
   ax1.legend(loc='upper right', fontsize=16)
   fig1.savefig('figS3b.pdf',bbox_inches='tight')
   plt.show()
