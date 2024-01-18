import numpy as np
from func_single import *
import matplotlib.pyplot as plt

def timeseries(dist, w, farray, t):
   # Computes the time-correlation function for time t

   F = np.zeros_like(t, dtype=complex)
   i = 0
   for f in farray:
      wf = f*w
      wtime = dist[i]*np.exp(-1j*wf*t)
      F = F + wtime
      i = i+1

   return(F)

def DipoleCorr(t, N, farraya, farrayb, wa, n, Sa, Sb, wbj, Omega, mu):
   # Given a harmonic oscillator with frequency wa and Huang-Rhys factor Sa and a collection of
   # n harmonic oscillators with frequency wbj and Huang-Rhys factors Sb and an electronic transition 
   # frequency Omega, this generates the dipole time-correlation function over an interval -T to T 

   Fa = np.zeros_like(t, dtype=complex)
   Fb = np.zeros((n,np.size(Fa)), dtype=complex)

   for f in farraya:
      # Franck-Condon factors
      FCa = FC_out(Sa, N, np.array([f]))
      wf = f*wa
      wtime = FCa**2*np.exp(-1j*wf*t)
      Fa = Fa + wtime

   for i in range(n):
      for f in farrayb:
         # Franck-Condon factors
         FCb = FC_out(Sb[i], 0, np.array([f]))
         wf = f*wbj[i]
         wtime = FCb**2*np.exp(-1j*wf*t)
         Fb[i] = Fb[i] + wtime

   F = Fa
   for i in range(n):
      F = np.multiply(F,Fb[i])

   F = mu**2*np.multiply(np.exp(-1j*Omega*t),F)

   return(F)

def DipoleCorrToProbDist(F, wseries, t, mu):
   # Computes the Fourier transform of F(t) and returns P(w)

   Nt = t.size

   P = np.zeros_like(wseries, dtype=complex)

   i = 0
   for wi in wseries:
      P[i] = np.sum(np.multiply(np.exp(1j*wi*t),F))/Nt
      i = i+1

   P = P.real/(mu**2)
   Ptot = np.sum(P)
#   print('DipCorrToProb',Ptot)

   return(P)

def ProbDistToA(P, wseries, mu):
   # Computes the absorption cross section from the probability distribution over frequency

   A = mu**2*np.multiply(P,wseries) 

   return(A)

def AToProbDist(A, wseries, mu):
   # Computes the probability distribution over frequency from absorption cross section A

   P = np.divide(A,wseries)/(mu**2)

   Ptot = np.sum(P)
#   print('AToProbDist',Ptot)

   return(P)

def ProbDistToDipoleCorr(P, mu, wseries, t):
   # Computes the dipole correlation F(t) from the Fourier transform of P(w)
   
   F = np.zeros_like(t, dtype=complex)
   
   i = 0
   for w in wseries:
      F = F + P[i]*np.exp(-1j*w*t)
      i = i+1

   F = mu**2*F

   return(F)
   
def TruncateAmulti(A, wseries, eta0, mu, Omega, wa):
   # Chooses a range of frequencies around Omega where the probability distribution is localized

   etamax = 1.2*(eta0)*mu**2*Omega
   Anew = A.copy()

   # Setting lower frequency limit
   i = 0
   for wi in wseries:
      if wi>=Omega-wa:
         Anew[:i]=0
         break
      elif A[i]>etamax:
         Anew[:i]=0
         break

      i = i+1


   # Setting higher frequency limit
   i = 0
   for wi in reversed(wseries):
      if wi<=Omega+wa:
         Anew[-i+1:]=0
         break
      elif A[-i]>etamax:
         if i > 1:
            Anew[-i+1:]=0
         break

      i = i+1

   return(Anew)

def ProbDistToS(dist, wseries, N, Omega, wa, dw):
   # Obtain the Huang-Rhys factor S from the divided probability distribution dist using the variance var=S(2N)
   
   farray = np.arange(-15,16)

   distnew = np.zeros_like(farray, dtype=float)

   i = 0
   j = 0
   for wi in wseries:
      if wi <= float(farray[j]) and wi+dw >= float(farray[j]):
         distnew[j] = max(dist[i],dist[i+1])
         j = j+1
         if j>=farray.size:
            break

      i = i + 1

   
   # Setting lower frequency limit
   i = 0
   for wi in farray:

      if distnew[i]>=0.07:
         distnew[:i]=0
         break
      if wi>=-wa:
         distnew[:i]=0
         break

      i = i+1


   # Setting higher frequency limit
   i = 0
   for wi in reversed(farray):
      if distnew[-i]>=0.07:
         if i >= 1:
            distnew[-i+1:]=0
            break

      if wi<=wa:
         distnew[-i+1:]=0
         break

      i = i+1

    
#   print(distnew)

   distnewtot = np.sum(np.abs(distnew))
   distnew = np.abs(distnew)/distnewtot


   mean = np.sum(np.multiply(distnew,farray))
   var = np.sum(np.multiply(distnew,np.square(farray-mean)))

   Sest = var/(2*N)

   return(Sest)

def NoiseFreeA(t, wseries, N1, farraya, farrayb, wa, n, Sa, Sb, wbj, Omega, mu):
   # Obtaining noise-free absorption cross section 
 
   #Dipole correlation function over a time interval -T to T
   F1 = DipoleCorr(t, N1, farraya, farrayb, wa, n, Sa, Sb, wbj, Omega, mu)

   # Fourier transform F(t) to get P(w)
   P1 = DipoleCorrToProbDist(F1, wseries, t, mu)

   # Get A(w) from P(w)
   A1 = ProbDistToA(P1, wseries, mu)
  
   return(A1)

def ParticularNoise1(t, wseries, mu, Omega, wa, A1, A0, etaA, etaA0, eta0, N1, dw):

   ######### Add noise to A(w) #############

   A1noisy = A1 + etaA
   A0noisy = A0 + etaA0

   ######### Truncate Anoisy(w) ############
   A1noisytrunc = TruncateAmulti(A1noisy, wseries, eta0, mu, Omega, wa)
   A0noisytrunc = TruncateAmulti(A0noisy, wseries, eta0, mu, Omega, wa)

   ######### Obtain Pnoisy(w) ##############
   P1noisy = AToProbDist(A1noisytrunc, wseries, mu)
   P0noisy = AToProbDist(A0noisytrunc, wseries, mu)

   ######### Fourier transform Pnoisy(w) to Fnoisy(t) #########
   F1noisy = ProbDistToDipoleCorr(P1noisy, mu, wseries, t)
   F0noisy = ProbDistToDipoleCorr(P0noisy, mu, wseries, t)

   ######### Divide Fnoisy(t) by F0noisy(t) to cancel contribution from other modes ########
   Fdiv1 = np.divide(F1noisy,F0noisy)

   ######### Fourier transform Fdiv(t) to obtain Pdiv(w) and Adiv(w) ###################
   Pdiv1 = DipoleCorrToProbDist(Fdiv1, wseries, t, 1)

   ######### Obtain Huang-Rhys factor #########
   Sest1 = ProbDistToS(Pdiv1, wseries, N1, Omega, wa, dw)
  
   return(Sest1,Pdiv1)

def ParticularNoise2(t, wseries, mu, Omega, wa, A1, A2, A0, etaA, etaA0, eta0, N1, N2, dw):

   ######### Add noise to A(w) #############

   A1noisy = A1 + etaA
   A2noisy = A2 + etaA
   A0noisy = A0 + etaA0

   ######### Truncate Anoisy(w) ############
   A1noisytrunc = TruncateAmulti(A1noisy, wseries, eta0, mu, Omega, wa)
   A2noisytrunc = TruncateAmulti(A2noisy, wseries, eta0, mu, Omega, wa)
   A0noisytrunc = TruncateAmulti(A0noisy, wseries, eta0, mu, Omega, wa)

   ######### Obtain Pnoisy(w) ##############
   P1noisy = AToProbDist(A1noisytrunc, wseries, mu)
   P2noisy = AToProbDist(A2noisytrunc, wseries, mu)
   P0noisy = AToProbDist(A0noisytrunc, wseries, mu)

   ######### Fourier transform Pnoisy(w) to Fnoisy(t) #########
   F1noisy = ProbDistToDipoleCorr(P1noisy, mu, wseries, t)
   F2noisy = ProbDistToDipoleCorr(P2noisy, mu, wseries, t)
   F0noisy = ProbDistToDipoleCorr(P0noisy, mu, wseries, t)

   ######### Divide Fnoisy(t) by F0noisy(t) to cancel contribution from other modes ########
   Fdiv1 = np.divide(F1noisy,F0noisy)
   Fdiv2 = np.divide(F2noisy,F0noisy)

   ######### Fourier transform Fdiv(t) to obtain Pdiv(w) and Adiv(w) ###################
   Pdiv1 = DipoleCorrToProbDist(Fdiv1, wseries, t, 1)
   Pdiv2 = DipoleCorrToProbDist(Fdiv2, wseries, t, 1)

   ######### Obtain Huang-Rhys factor #########
   Sest1 = ProbDistToS(Pdiv1, wseries, N1, Omega, wa, dw)
   Sest2 = ProbDistToS(Pdiv2, wseries, N2, Omega, wa, dw)
   
   return(Sest1,Sest2,Pdiv1,Pdiv2)

def NoiseAverage(t, wseries, mu, Omega, wa, A1, A2, A0, eta0, N1, N2, dw, Nnoise, farraya, flag2):

   Svals1 = np.zeros(Nnoise)
   Svals2 = np.zeros(Nnoise)

   # Add noise to A(w)
   k = 0
   for ni in range(Nnoise):   ########## Loop over noise
      print('Noise',ni)
      etaA = mu**2*Omega*noise(wseries, eta0)
      etaA0 = mu**2*Omega*noise(wseries, eta0)

      i = 0
      for wi in wseries:
         if wi<=Omega+min(farraya)*wa:
            etaA[i] = 0
            etaA0[i] = 0
         i = i+1
      
      if flag2 == 1:
         Svals1[k],Pdiv1 = ParticularNoise1(t, wseries, mu, Omega, wa, A1, A0, etaA, etaA0, eta0, N1, dw)
      elif flag2 == 2:
         Svals1[k],Svals2[k],Pdiv1,Pdiv2 = ParticularNoise2(t, wseries, mu, Omega, wa, A1, A2, A0, etaA, etaA0, eta0, N1, N2, dw)
      k = k + 1

   meanS1 = np.mean(Svals1)
   stdS1 = np.std(Svals1)
   meanS2 = np.mean(Svals2)
   stdS2 = np.std(Svals2)

   if flag2 == 1:
      return(meanS1,stdS1)
   elif flag2 == 2:
      return(meanS1,stdS1,meanS2,stdS2)
