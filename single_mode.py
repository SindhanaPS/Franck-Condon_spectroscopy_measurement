import numpy as np
from func_single import *
import math as m
import matplotlib.pyplot as plt

########################################################
####### flag1 = 0     Generate data for Fig. 1d-f
####### flag1 = 1     Generate data for Fig. 2
####### flag1 = 2     Generate data for Fig. S1
########################################################

flag1 = 2
flag2 = 2

S = 0.5
mu = 1
N = 10
Omega = 20
wa = 1

beta = -m.log(N/(1+N))

farray = np.arange(-15,16)

FC = FC_out(S, N, farray)

therm = thermal(S, beta, farray)
fock = np.square(FC)

FC = FC_out(S, 0, farray)
fock0 = np.square(FC)

################ check ##################

#meantherm = np.sum(np.multiply(therm,farray))
#meanfock = np.sum(np.multiply(fock,farray))
#vartherm = np.sum(np.multiply(therm,np.square(farray-meantherm)))
#varfock = np.sum(np.multiply(fock,np.square(farray-meanfock)))

#print(meantherm,meanfock,S)
#print(vartherm,varfock,(2*N+1)*S)

#eta0 = 0.1

#dist = fock
#A = AFromProbDist(dist, farray, mu, Omega)
#eta = mu**2*Omega*noise(farray, eta0)             # Generate noise
#Anoisy = A + eta 
#Atrunc = TruncateA(Anoisy, farray, eta0, mu, Omega)       # Truncate noisy absorption cross section to the relevant region

#fig, ax = plt.subplots(3)
#ax[0].bar(farray,A)
#ax[1].bar(farray,Atrunc)
#ax[2].bar(farray,Anoisy)

#plt.show()

#distnew = ProbDistFromA(Atrunc, farray, mu, Omega)   # Obtain the probability distribution from the truncated absorption cross section
#Svals = SFromProbDistVar(distnew, farray, N)

#print(Svals)

################################################



############### Generate data for Fig. S1  ###############

if flag1 == 2:
   S = np.logspace(np.log10(0.00001),np.log10(0.5),num=10)
   n = 5000
   eta0 = 0.1

   meanDd0,stdDd0 = Ddarray(S, 0, 0, farray, n, beta, mu, Omega, eta0)
   meanDdtherm,stdDdtherm = Ddarray(S, N, 1, farray, n, beta, mu, Omega, eta0)
   meanDdfock,stdDdfock = Ddarray(S, N, 0, farray, n, beta, mu, Omega, eta0)
   meanDdcoh,stdDdcoh = Ddarray(S, N, 2, farray, n, beta, mu, Omega, eta0)

   fname = 'EstimateDd.txt'

   np.savetxt(fname, np.transpose([np.sqrt(2*S), meanDd0, stdDd0, meanDdtherm, stdDdtherm, meanDdfock, stdDdfock, meanDdcoh, stdDdcoh]))

############## Generate data for Fig. 2 ############

if flag1 == 1:

   S = np.logspace(np.log10(0.00001),np.log10(0.06),num=100)
   n = 5000

   if flag2 == 0:
      eta0 = 0.1
   elif flag2 == 1:
      eta0 = 0.1/4
   elif flag2 == 2:
      eta0 = 0.1/9

   numN = 400

   np.savetxt('Dd.txt', np.sqrt(2*S))

   i = 0
   for si in S:
      f = open(f'Dd_{i}.txt', 'r+')
      f.truncate(0) # need '0' when using r+
      f.close()
      i = i + 1

   for Ni in np.logspace(2,10,num=10-2+1,base=2, dtype=int):
      meanDdfock,stdDdfock = Ddarray(S, Ni, 0, farray, n, beta, mu, Omega, eta0)
      i = 0
      for si in S:
         f = open(f'Dd_{i}.txt','a')
         np.savetxt(f, [Ni,meanDdfock[i],stdDdfock[i]], newline=" ")
         i = i + 1
         f.write('\n')
         f.close()

################# Generate data for Fig 1d-f ###################

if flag1 == 0:

   S1 = np.square(1)/2
   S2 = np.square(0.3)/2
   eta0 = 0.1

   farray = np.arange(-7,8)
   wseries = Omega + wa*farray

   FC01 = FC_out(S1, 0, farray) 
   FC0 = FC_out(S2, 0, farray) 
   FC8 = FC_out(S2, 8, farray) 

   fock01 = np.square(FC01)
   fock0 = np.square(FC0)
   fock8 = np.square(FC8)

   A01 = AFromProbDist(fock01, farray, mu, Omega)
   A0 = AFromProbDist(fock0, farray, mu, Omega)
   A8 = AFromProbDist(fock8, farray, mu, Omega)

   eta = mu**2*Omega*noise(farray, eta0)             # Generate noise
   Anoisy01 = A01 + eta 
   Anoisy0 = A0 + eta 
   Anoisy8 = A8 + eta 

   np.savetxt('spectrum_N0_single_large.txt', np.transpose([wseries, Anoisy01]))
   np.savetxt('spectrum_N0_single.txt', np.transpose([wseries, Anoisy0]))
   np.savetxt('spectrum_N8_single.txt', np.transpose([wseries, Anoisy8]))

