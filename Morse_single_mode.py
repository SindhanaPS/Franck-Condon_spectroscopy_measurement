import numpy as np
from func_single import *
import math as ma
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre
from scipy.constants import hbar, m_u, c

# Morse potential parameters
flag1 = 0
############################### Parameters ####################################
#### flag1 = 0    -    Parameters for CaH
#### flag1 = 1    -    Parameters for SrF 
###############################################################################

if flag1 == 0:
   w_e = 1298.34 # frequency in cm^-1
   w_e_X_e = 19.1 # frequency in cm^-1
   r_e = 2.0025  # Equilibrium bond length in Angstrom
   deltad = -0.0285              # in Å

   # Masses of the atoms - CaH
   m_Ca = 40.078 * m_u  # Mass of Calcium atom in kg
   m_H = 1.00784 * m_u  # Mass of Hydrogen atom in kg

   # Reduced mass
   mu = (m_H * m_Ca) / (m_H + m_Ca)

   fname = 'CaH.txt'
   fname2 = 'CaH_M2-M12_approx.txt'
   fname3 = 'CaH_M2-M12.txt'
   msize = 8

elif flag1 == 1:
   w_e = 502.4 # frequency in cm^-1
   w_e_X_e = 2.2 # frequency in cm^-1
   r_e = 2.075  # Equilibrium bond length in Angstrom
   deltad = 0.00473              # in Å

   # Masses of the atoms - SrF
   m_Sr = 87.62 * m_u  # Mass of Strontium atom in kg
   m_F = 18.99 * m_u  # Mass of Fluorine atom in kg

   # Reduced mass
   mu = (m_Sr * m_F) / (m_Sr + m_F)

   fname = 'SrF.txt'
   fname2 = 'SrF_M2-M12_approx.txt'
   fname3 = 'SrF_M2-M12.txt'
   msize = 30

hOmega = 20        # electronic transition frequency in units of w_e_rad
muD = 1            # transition dipole in reduced units

fmax = 21
# Conversion constants
eV_to_J = 1.60218e-19
c_cm_per_s = c * 100  # Speed of light in cm/s
cm_inv_to_eV = 1.2398*10**(-4)

# Convert w_e to angular frequency (rad/s)
w_e_rad = w_e * c_cm_per_s * 2 * np.pi

# Calculate D_e and a
D_e_cm = w_e**2/(4*w_e_X_e)  # Dissociation energy in cm^-1
D_e = D_e_cm*cm_inv_to_eV  # Dissociation energy in eV
a = np.sqrt(w_e_rad**2 * mu / (2 * D_e * eV_to_J)) * 10**(-10)  # Convert from m^-1 to Å^-1

# Calculate Lambda
Lambda = 2 * D_e * eV_to_J / (hbar * w_e_rad)

########## Plot Morse potential and wavefunction ###########
r = np.linspace(1.4, 10, 5000)
potential = morse_potential(r, D_e, a, r_e)/(hbar*w_e_rad)

plt.plot(r, potential, 'k-')

ax = plt.gca()

for i in range(ma.floor(Lambda-0.5)+1):
   Ei = morse_energy(i, Lambda, w_e_rad)/(hbar*w_e_rad)
   x1, x2 = find_classical_turning_point(r, D_e, a, r_e, Ei*hbar*w_e_rad)
   plt.hlines(y=Ei, xmin=x1, xmax=x2, color='k', linestyle='-')

   if i == 10:
      wavefunc = morse_wavefunction(i, r, Lambda, a, r_e)
      plt.plot(r, Ei+0.4*wavefunc, 'r-')

plt.xlabel('Bond Length (Å)')
plt.ylabel(r'E/$\hbar\omega_a$')

plt.savefig('Morse.pdf',bbox_inches='tight')

wavefunc = morse_wavefunction(8, r, Lambda, a, r_e)

###########################################################

r = np.linspace(0.05, 20, 60000)
potential = morse_potential(r, D_e, a, r_e)

farray = np.arange(-fmax,fmax)
meanEarr = np.zeros(msize,dtype=float)
varEarr = np.zeros(msize,dtype=float)
for m in range(msize):
   Em = morse_energy(m, Lambda, w_e_rad)/(hbar*w_e_rad)
   Earray = morse_energy(m+farray, Lambda, w_e_rad)/(hbar*w_e_rad)
   FC = FC_Morse(deltad, m, farray, r, Lambda, a, r_e)
   FC2 = np.square(FC)
   meanEarr[m] = np.sum(np.multiply(FC2,Earray-Em))
   varEarr[m] = np.sum(np.multiply(FC2,np.square(Earray-Em)))
   print(m,sum(FC2))
#   varEarr[m] = varEarr[m]/sum(FC2)
#   plt.bar(farray,FC2)
#   plt.show()
#   print(meanE,varE)

print(f"De = {D_e} eV")
print(f"re = {r_e} Å^-1")
print(f"we = {w_e} cm^-1")
print(f"a = {a} Å^-1")
print(f"Dd = {deltad} Å")
print(f"Lambda = {Lambda}")

deltad_red = deltad * np.sqrt(mu*w_e_rad/hbar)*10**(-10)
print(f"Dd* = {deltad_red}")
S = w_e/w_e_X_e
print(f"S = {S}")

n= np.linspace(0,msize,100)
u = n+0.5
fS = u+(-3*np.square(u)+1.25)/S+(2*np.power(u,3)-2.5*u)/S**2

np.savetxt(fname2, np.transpose([n,fS*deltad_red**2]))
np.savetxt(fname3, np.transpose([np.arange(msize), varEarr-np.square(meanEarr)]))

meanDd = np.zeros(msize)
stdDd = np.zeros(msize)
percDd = np.zeros(msize)

for m in range(msize):
   meanDd[m],stdDd[m] = DdMorse(deltad, m, farray, r, Lambda, a, r_e, w_e_rad, S, D_e, 400, muD, hOmega, 0.05)
   percDd[m] = abs(abs(deltad)-meanDd[m])*100/abs(deltad)
   print(m,meanDd[m],stdDd[m],percDd[m])

np.savetxt(fname, np.transpose([np.arange(msize), meanDd, stdDd, percDd]))
