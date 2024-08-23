import numpy as np
import math as ma
from scipy.constants import hbar, m_u, c
from scipy.special import eval_genlaguerre, gamma, gammaln

def FC_out(S, m, farray):
    #The function FC_out computes <m|m+f>' where |>' is displaced from |> with
    #Huang-Rhys factor S and m is small. f can also be an array

    FC = np.zeros_like(farray,dtype=float)

    for i in range(farray.size):
       f = farray[i]
       if m + f < 0:
           FC[i] = 0
       else:
           s = 0
           j = int(max(0, f))
           sterm = 0

           while True:
               sterm = (-S) ** j * ma.comb(m, m + int(f) - j) / ma.factorial(j)
               s += sterm
               j += 1

               if m + f - j < 0:
                   break

               if abs(sterm) <= 0.000000000000001 * abs(s) and sterm != 0 and j <= m + f:
                   break

           FC[i] = (-1.0) ** f * ma.sqrt(ma.factorial(m + int(f)) / ma.factorial(m)) * ma.sqrt(ma.exp(-S) / S ** f) * s

    return FC

def thermal(S, beta, farray):
   # The function thermal computes the coefficient of the peaks delta(w-W-fw_v) starting 
   # from an initial thermal state with inverse temperature beta
   x = ma.exp(-beta)
   N = x/(1-x)

   s = np.zeros_like(farray, dtype=float)
   nu = 0
   avgnu = 0
   while True:
      if beta !=0:
         rhonu = ma.exp(-nu*beta)*(1-ma.exp(-beta))               # population of state |nu> at inverse temperature beta
      else:
         if nu == 0:
            rhonu = 1
         else:
            rhonu = 0

      avgnu += rhonu*nu

      FC = FC_out(S, nu, farray)

      sterm = rhonu*np.square(FC)
      s += sterm
      ch = s[np.nonzero(np.abs(s))]
      if np.max(np.abs(sterm))<0.000000001:
         break
      nu = nu +1

   return(s)

def coherent(S, N, farray):
   # This function computes \sum_n p_n |<n|n+f>'|^2

   alpha0 = np.sqrt(N)
   stddev = int(np.abs(alpha0))
   mean = int(np.abs(alpha0)**2)

   if mean-3*stddev>=0:
      lower = mean-3*stddev
   else:
      lower = 0

   upper = mean + 3*stddev

   # probability of bein in n, p_n
   p = np.zeros(upper-lower+1)
   n = np.arange(lower,upper+1)

   for k in range(n.size):
      p[k] = np.exp(-mean)*np.power(float(mean),n[k])/ma.factorial(n[k])

   FC2n = np.zeros((n.size,farray.size))
   total = np.zeros_like(farray)

   for k in range(n.size):
      FC2n[k] = p[k]*np.square(FC_out(S,n[k],farray))
      total = total + FC2n[k]

   return total

def AFromProbDist(dist, farray, mu, Omega):
   # Obtain the absorption cross section from the probability distribution

   omega = Omega+farray

   A = mu**2*np.multiply(dist,omega)

   return(A)

def SFromProbDistVar(dist, farray, N):
   # Obtain the Huang-Rhys factor S from the probability distribution dist using the variance var=S(2N+1)

   mean = np.sum(np.multiply(dist,farray))
   var = np.sum(np.multiply(dist,np.square(farray-mean)))

   Sest = var/(2*N+1)

   return(Sest)

def noise(farray, eta0):
   # Returns uniform random noise between 0 and eta0 to all the frequency components (white noise) of the absorption cross section

   # white noise
   eta = np.random.uniform(low=0, high=eta0, size=farray.size)

   return(eta)

def ProbDistFromA(A, farray, mu, Omega):
   # Obtains the probability distribution from the absorption cross-section and normalizes it

   omega = Omega+farray
   P = np.divide(A,omega)/(mu**2)

   Ptot = np.sum(P)
   P = P/Ptot

   return(P)

def TruncateA(A, farray, eta0, mu, Omega):
   # Chooses a range of frequencies around Omega where the probability distribution is localized

   etamax = eta0*mu**2*Omega
   Anew = A.copy()

   # Setting lower frequency limit
   i = 0
   for f in farray:
#      if f>=-1:
      if f>=0:
         Anew[:i]=0
         break
      elif A[i]>etamax:
         Anew[:i]=0
         break

      i = i+1


   # Setting higher frequency limit
   i = 0
   for f in reversed(farray):
#      if f<=1:
      if f<=0:
         Anew[-i+1:]=0
         break
      elif A[-i]>etamax:
         if i > 1:
            Anew[-i+1:]=0
         break
      
      i = i+1

   return(Anew)

def Ddarray(S, N, fnum, farray, n, beta, mu, Omega, eta0):
   # Dd/2 = S^2 
   
   meanDd = np.zeros_like(S)
   stdDd = np.zeros_like(S)

   i = 0
   for s in S:
      Svals = np.zeros(n)

      if fnum == 0:
         FC = FC_out(s, N, farray)
         fock = np.square(FC)
         dist = fock
      elif fnum == 1:
         therm = thermal(s, beta, farray)
         dist = therm
      elif fnum == 2:
         coh = coherent(s, N, farray)
         dist = coh
      
      A0 = AFromProbDist(dist, farray, mu, Omega)

      for ni in range(n):
         eta = mu**2*Omega*noise(farray, eta0)                            # Generate noise
         Anoisy = A0 + eta                                    # Add noise to absorption cross section
         Atrunc = TruncateA(Anoisy, farray, eta0, mu, Omega)  # Truncate noisy absorption cross section to the relevant region
         distnew = ProbDistFromA(Atrunc, farray, mu, Omega)   # Obtain the probability distribution from the truncated absorption cross section
         Svals[ni] = SFromProbDistVar(distnew, farray, N)     # Obtain the Huang-Rhys factor from the variance for this realization of noise
      
      meanDd[i] = np.mean(np.sqrt(2*Svals))
      stdDd[i] = np.std(np.sqrt(2*Svals))
      i = i+1

   return(meanDd,stdDd)

def morse_potential(r, D_e, a, r_e):
    # Morse potential
    eV_to_J = 1.60218e-19
    return D_e * eV_to_J * (1 - np.exp(-a * (r - r_e)))**2

def morse_energy(n, Lambda, w_e_rad):
    if isinstance(n, np.ndarray) == True:
       E = np.zeros(n.shape,dtype=float)
       #print(n)
       for i in range(n.size):
          if n[i]>=0 and n[i]<=ma.floor(Lambda-0.5):
             E[i] = ((n[i] + 1/2) - 1 / (2 * Lambda) * (n[i] + 1/2)**2)* hbar * w_e_rad
             #print(n[i],E[i])
          else:
             E[i] = 0
    else:
       if n>=0 and n<=ma.floor(Lambda-0.5):
          E = ((n + 1/2) - 1 / (2 * Lambda) * (n + 1/2)**2) * hbar * w_e_rad
       else:
          E = 0
    return E

def find_classical_turning_point(r, D_e, a, r_e, Ei):
    pot = morse_potential(r, D_e, a, r_e)
  
    xmin = 0
    xmax = 0 
    for j in range(r.size-1):
       if pot[j]>Ei and pot[j+1]<=Ei:
          xmin = r[j+1]

       if pot[j]<=Ei and pot[j+1]>Ei:
          xmax = r[j]

       if j == r.size-2 and xmin == 0:
          xmin = r[0]
       if j == r.size-2 and xmax == 0:
          xmax = r[j+1]
    
    return (xmin,xmax)

def morse_wavefunction(n, r, Lambda, a, r_e):
    # Function to calculate Morse wavefunctions

    if n>=0 and n<=ma.floor(Lambda-0.5):
       z = 2 * Lambda * np.exp(-a * (r - r_e))
       #N_n = np.sqrt(ma.factorial(n) * (2 * Lambda - 2 * n - 1)*a  / gamma(2 * Lambda - n))
       #wavefunc = N_n * z**(Lambda - n - 0.5) * np.exp(-0.5 * z) * eval_genlaguerre(n, 2 * Lambda - 2 * n - 1, z)
       N_n = np.sqrt(ma.factorial(n) * (2 * Lambda - 2 * n - 1)*a)
       log_wavefunc = (Lambda - n - 0.5) * np.log(z) - 0.5 * z-0.5*gammaln(2 * Lambda - n)
       wavefunc = N_n*np.exp(log_wavefunc)*eval_genlaguerre(n, 2 * Lambda - 2 * n - 1, z)
    else:
       wavefunc = np.zeros_like(r)
    return wavefunc

def overlap(f1,f2,dr):
    return sum(np.multiply(f1, f2))*dr

def FC_Morse(deltad, m, farray, r, Lambda, a, r_e):
    #The function FC_Morse computes <m|m+f>' where |>' is displaced from |> by deltad and they are eigenstates
    #of the Morse potential

    dr = r[1]-r[0]
    FC = np.zeros_like(farray,dtype=float)

    wavefuncm = morse_wavefunction(m, r, Lambda, a, r_e)

    for i in range(farray.size):
       f = farray[i]
       if m + f <0:
          FC[i] = 0
       else:
          wavefunci = morse_wavefunction(m + farray[i], r, Lambda, a, r_e + deltad)
          FC[i] = overlap(wavefuncm,wavefunci,dr)

    return FC

def AFromProbDistMorse(dist, omegaArr, mu, hOmega):
   # Obtain the absorption cross section from the probability distribution

   omega = hOmega+omegaArr

   A = mu**2*np.multiply(dist,omega)

   return(A)

def ProbDistFromAMorse(A, omegaArr, mu, hOmega):
   # Obtains the probability distribution from the absorption cross-section and normalizes it

   omega = hOmega+omegaArr
   P = np.divide(A,omega)/(mu**2)

   Ptot = np.sum(P)
   P = P/Ptot

   return(P)

def DeltadFromProbDistVar(dist, omegaArr, n, D_e, w_e_rad, a, S):
   # Obtain Delta d from the probability distribution dist using the variance
  
   eV_to_J = 1.60218e-19 
   meanEarr = np.sum(np.multiply(dist,omegaArr))
   varEarr = np.sum(np.multiply(dist,np.square(omegaArr)))

   u = n+0.5
   fS = 2*u/S+(-6*np.square(u)+2.5)/S**2+(4*np.power(u,3)-5*u)/S**3
   Deltad = np.sqrt((varEarr-np.square(meanEarr))/fS)/(2*a*(D_e*eV_to_J/(hbar*w_e_rad)))

   return(Deltad)

def DdMorse(deltad, m, farray, r, Lambda, a, r_e, w_e_rad, S, D_e, n, mu, hOmega, eta0):

   Ddvals = np.zeros(n)

   Em = morse_energy(m, Lambda, w_e_rad)/(hbar*w_e_rad)
   Earray = morse_energy(m+farray, Lambda, w_e_rad)/(hbar*w_e_rad)
   FC = FC_Morse(deltad, m, farray, r, Lambda, a, r_e)
   dist = np.square(FC)
   A0 = AFromProbDistMorse(dist, Earray-Em, mu, hOmega)

#   Ddvals[0] = DeltadFromProbDistVar(dist, Earray-Em, m, D_e, w_e_rad, a, S)                  # Obtain the Delta d from the variance for this realization of noise
#   print(Ddvals[0])
   for ni in range(n):
      eta = mu**2*hOmega*noise(farray, eta0)                            # Generate noise
      Anoisy = A0 + eta                                                 # Add noise to absorption cross section
      Atrunc = TruncateA(Anoisy, farray, eta0, mu, hOmega)              # Truncate noisy absorption cross section to the relevant region
      distnew = ProbDistFromAMorse(Atrunc, Earray-Em, mu, hOmega)       # Obtain the probability distribution from the truncated absorption cross section
      Ddvals[ni] = DeltadFromProbDistVar(distnew, Earray-Em, m, D_e, w_e_rad, a, S)                  # Obtain the Delta d from the variance for this realization of noise

   meanDd = np.mean(Ddvals)
   stdDd = np.std(Ddvals)

   return(meanDd,stdDd)
