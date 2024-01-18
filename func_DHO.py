import math
import numpy as np
import math as m
from cmath import e,pi
from scipy.special import hermite

def howave(x,n):
    h1=hermite(n)
    return (1/np.sqrt(pow(2,n)*m.factorial(n)))*(1/pi)**(1/4)*h1(x)*e**(-x**2/2)

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
               sterm = (-S) ** j * math.comb(m, m + int(f) - j) / math.factorial(j)
               s += sterm
               j += 1

               if m + f - j < 0:
                   break

               if abs(sterm) <= 0.00000000001 * abs(s) and sterm != 0 and j <= m + f:
                   break

           FC[i] = (-1) ** f * math.sqrt(math.factorial(m + int(f)) / math.factorial(m)) * math.sqrt(math.exp(-S) / S ** f) * s

    return FC
