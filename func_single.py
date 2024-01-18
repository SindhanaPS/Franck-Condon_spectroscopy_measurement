import numpy as np
import math as ma

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

