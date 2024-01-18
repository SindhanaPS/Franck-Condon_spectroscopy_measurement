import numpy as np
import matplotlib.pyplot as plt
import math as m
import  matplotlib
from cmath import e,pi
from func_DHO import *
from matplotlib import rc,cm
from matplotlib import rcParams

flag1 = 2

##################################################################################
########## flag1 == 0     Fig. 1a
########## flag1 == 1     Fig. 1b
########## flag1 == 2     Fig. 1c
###################################################################################

if flag1 == 0:
   d = 1           #displacement
   ni = 0
   fname = 'fig1_a.pdf'
elif flag1 == 1:
   d = 0.3           #displacement
   ni = 0
   fname = 'fig1_b.pdf'
elif flag1 == 2:
   d = 0.3           #displacement
   ni = 8
   fname = 'fig1_c.pdf'

###################################################################################

###################################################################################
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 15}

rc('font', **font)

rcParams['text.latex.preamble'] = [
       r'\usepackage{physics}',
       r'\usepackage{amsmath}',
]

plt.rcParams['text.usetex'] = True
###################################################################################


E2 = 15         #min energy of second oscillator
w = 1

x1 = np.linspace(-5,5,200)
x2 = np.linspace(-5+d,5+d,200)

#Potential energy
V1 = 0.5*x1**2
V2 = 0.5*(x2-d)**2 + E2

#scale
scale = 2

#Initial wavefunction
Ei = ni+0.5
ximax = np.sqrt(2*Ei)
delta = 2
xi = np.linspace(-(ximax+delta),ximax+delta,200)
yi = scale*howave(xi,ni)

#Final wavefunctions 
N = 11                                # number of states
Ef = [i+0.5 for i in range(N)]
xfmax = np.sqrt(2*Ef)
xf = [np.linspace(-(xfmax[i]+delta)+d,(xfmax[i]+delta)+d,200) for i in range(N)]
yf = [ scale*howave(xf[i]-d,i) for i in range(N)]

#FC factors
S = d**2/2
f = np.arange(float(-ni),float(N-ni))
FC = FC_out(S,ni,f)
FC2 = np.square(FC)

ax = plt.gca()
#Plot wavefunctions
rightend = min(x1)-delta
cmap = matplotlib.cm.get_cmap('Purples')
norm = matplotlib.colors.LogNorm(vmin=10**(-7), vmax=1.0)
plt.plot(xi,yi+Ei,color='lime',linewidth=2)

for k in range(N):
   i = N-k-1
   if i%1==0:
      c = cmap(FC2[i])
      plt.plot(xf[i],yf[i]+Ef[i]+E2,color=c,linewidth=2)
      if i%2==0:
         ax.text(rightend+0.3,Ef[i]+E2,r"$\mathbf{j}$".replace('j',str(i)),fontsize=15,fontweight='bold')

for i in range(N):
   if i%1==0:
      plt.hlines(Ef[i]+E2,rightend,5+d,colors='gray',linestyle='dashed',linewidth=0.25)

#Plot FC factor subplot
for i in range(N):
   leftend = rightend - 3*FC2[i]
   plt.hlines(Ef[i]+E2,leftend,rightend,colors='orange',linewidth=4)

Emax = E2 + 13
plt.arrow(rightend,E2,0,Emax-E2,color='black',head_width=0.4,head_length=0.4)
plt.arrow(rightend,E2,-3,0,color='black',head_width=0.8,head_length=0.2)
ax.text(rightend+0.2,Emax+0.5,r"$\mathbf{m}$",fontsize=15,fontweight='bold')

#Plot potential energy
plt.ylim(-1,10)
plt.plot(x1,V1,'black')
plt.ylim(-1,30)
plt.plot(x2,V2,'black')
#plt.colorbar(cm.ScalarMappable(cmap='Purples'), ax = [ax], shrink=0.3,orientation='horizontal',anchor=(0.65,-0.2),location='top')
#ax.text(5+d,Emax+3,r"$\mathbf{|\langle j |n\rangle'|^2}$".replace('j',str(ni)),fontsize=15)

plt.arrow(0,-0.7,d,0,head_width=0.4,head_length=0.1)
plt.arrow(d,-0.7,-d,0,head_width=0.4,head_length=0.1)
#ax.text(-0.5,-3.5,r"$\Delta d=j\sqrt{\frac{\hbar}{m\omega_a}}$".replace('j',str(d)),fontsize=30)

plt.vlines([0,d],0,Emax,colors='gray',linestyles='dashed',linewidth=0.25)
ax.text(rightend-2.5,E2-1.8,r"$\mathbf{|\langle j | m\rangle'|^2}$".replace('j',str(ni)),fontsize=15)
ax.set_aspect(0.5)
plt.ylim(-1,30)
plt.axis('off')
plt.savefig(fname,bbox_inches='tight',transparent=True)
plt.show()
