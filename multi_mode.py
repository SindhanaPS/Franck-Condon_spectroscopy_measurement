import numpy as np
from func_single import *
from func_multi import *
import math as m
import matplotlib.pyplot as plt

mu = 1
wa = 1
Omega = 20

########### Read a set of mode frequencies and Huang-Rhys factors from a file

data = np.loadtxt('wbj_Sb.txt')
wbj = data[:,0]
Sb = data[:,1]
n = Sb.size

##################################################################

T = 10*2*m.pi/min(wa,np.min(wbj))
dt = 2*m.pi/(10*Omega)

farraya = np.arange(-15,16)
farrayb = np.arange(0,6)

# Time series
t = np.linspace(-T,T,num=int(2*T/dt))

# Frequency series
dw = m.pi/T
Wmax = Omega + 2*np.max(farraya)*wa
Wmin = 2*np.min(farraya)*wa
wseries = np.linspace(Wmin,Wmax,num=int((Wmax-Wmin)/dw))

eta0 = 0.001
Sa = 0.02
N1 = 1
N2 = 10

# Noise free spectra
A0 = NoiseFreeA(t, wseries, 0, farraya, farrayb, wa, n, Sa, Sb, wbj, Omega, mu)
A1 = NoiseFreeA(t, wseries, N1, farraya, farrayb, wa, n, Sa, Sb, wbj, Omega, mu)
A2 = NoiseFreeA(t, wseries, N2, farraya, farrayb, wa, n, Sa, Sb, wbj, Omega, mu)

# Add noise to A(w) 
etaA = mu**2*Omega*noise(wseries, eta0) 
etaA0 = mu**2*Omega*noise(wseries, eta0) 

i = 0
for wi in wseries:
   if wi<=Omega+min(farraya)*wa:
      etaA[i] = 0
      etaA0[i] = 0
   i = i+1

Sest1,Sest2,Pdiv1,Pdiv2 = ParticularNoise2(t, wseries, mu, Omega, wa, A1, A2, A0, etaA, etaA0, eta0, N1, N2, dw)
np.savetxt('spectrum_N0.txt', np.transpose([wseries, A0 + etaA0]))
np.savetxt('spectrum_N1.txt', np.transpose([wseries, A1 + etaA]))
np.savetxt('spectrum_N10.txt', np.transpose([wseries, A2 + etaA]))
np.savetxt('spectrum_div_N1.txt', np.transpose([wseries, Pdiv1]))
np.savetxt('spectrum_div_N10.txt', np.transpose([wseries, Pdiv2]))
print(Sa,Sest1,Sest2)
