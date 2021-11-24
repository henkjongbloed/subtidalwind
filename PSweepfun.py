import numpy as np
import numpy.ma as ma
import matplotlib as mp
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors as co
#import matplotlib.colors as colors
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from generalFun import *
from matplotlib import ticker

class ParameterSweep:
    def __init__(self, gp, nondimDict, dim):
        self.gp = gp
        self.nondimDict = nondimDict # Governing parameters
        self.dim = dim               # Dimensional or non-dimensional
        self.nps = nondimDict['nps']
        self.gp['Class'] = 'PS'
        self.Ra = nondimDict['Ra']
        self.RaOri = np.copy(self.Ra)
        self.Fr = nondimDict['Fr']
        self.FrOri = np.copy(self.Fr)
        self.Fw = nondimDict['Fw']
        self.FwOri = np.copy(self.Fw)
        self.Sc = nondimDict['Sc']
        self.n = nondimDict['n']
        self.name = nondimDict['name']
        #print(np.amax(self.Fw))
        return
    
    def run(self):
        self.initPS()                  # Contstruct b and c, b0, b0, b00, c00
        self.solvePS()                 # Compute all relevant quantities (masks, )
        return self

    def initPS(self):
        self.a, self.b, self.c, self.d, self.b0, self.c0, self.Sb_X0, self.Phi_0, self.Phi_c, self.Phi_s, self.Sb_0, self.Xs, self.rs,self.Xs0, self.rs0,self.Xs1, self.rs1, self.D2, self.We0, self.M0 = np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n),np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n)
        self.Exx, self.Exy = np.empty(shape = (self.n, 3)), np.empty(shape = (self.n, 3))
        self.L, self.T, self.Reg, self.LsT = np.empty(shape = (self.n, 8)), np.empty(shape = (self.n, 8)), np.empty(shape = (self.n, 3)), np.empty(shape = (self.n, 8))
        self.mask, self.maskOri = np.full((self.n,8), False, dtype = bool), np.full((self.n,8), False, dtype = bool)
        self.maxUN = np.empty(shape = (self.n, 2))
        return
    
    def solvePS(self):
        if self.gp['mixAway']:
            i = 0
            fac = self.gp['m']
            #incr = False
            #mixPar = initMixPar(gp)
            nonunique = 0
            nonphysical = 0
            while i < self.n:
                #j = 0
                self.L[i,:], self.a[i], self.b[i], self.c[i], self.d[i], self.b0[i], self.c0[i] = nonDim2ODECoeff(self.gp, self.Ra[i], self.Fr[i], self.Fw[i], self.Sc)
                self.D2[i], self.Exx[i,:], self.Exy[i,:] = computeLocalExtrema(self.a[i]/self.d[i], self.b[i]/self.d[i], self.c[i]/self.d[i], 0.0)
                self.Sb_X0[i], self.mask[i,0] = solveCubic(np.array([self.a[i],self.b0[i],self.c0[i], -self.d[i]]))
                self.Sb_0[i], self.Phi_0[i] = processBC(self.gp, self.a[i], self.b[i], self.c[i], self.d[i], self.Ra[i], self.Fr[i], self.Fw[i], self.Sc, self.Sb_X0[i])
                self.rs[i], self.Xs[i], self.rs0[i], self.Xs0[i], self.rs1[i], self.Xs1[i], self.mask[i,1] = computersXs(self.gp, self.a[i], self.b[i], self.c[i], self.d[i], self.Sb_X0[i], self.Fr[i], self.Ra[i], self.Fw[i])
                self.mask[i,2] = computeNU(self.D2[i], self.Exx[i,:], self.Sb_X0[i], self.rs[i])
                #self.mask[i,2] = computeNU(self.gp,  self.a[i], self.b[i], self.c[i], self.d[i],  self.Sb_0[i])
                if nonunique == 0: #Save the original mask for plotting later on.
                    self.maskOri[i,0:3] = self.mask[i,0:3]
                if not any(self.mask[i,0:3]): #Solution exists and is unique. Possibly unstable stratification / negative salt.
                    self.mask[i,3], self.mask[i,4], self.maxUN[i,:], self.Phi_c[i], self.Phi_s[i] = computePhysicalMasksPS(self.gp, self.a[i], self.b[i], self.c[i], self.d[i], self.Ra[i], self.Fr[i], self.Fw[i], self.Sc, self.Sb_X0[i], self.Sb_0,self.rs[i])
                    if nonphysical == 0: #save original result for eventual masks
                        self.maskOri[i,3:5] = self.mask[i,3:5]
                        #self.maxUN[i,:] = maxUN
                    if any(self.mask[i,3:5]):
                        nonphysical =+ 1
                        self.Ra[i], self.Fw[i], self.mask[i,6] = findMixing(fac, self.Ra[i], self.Fw[i])
                        #print(str(i/self.n) + ': PhysMask')
                        continue
                else: #Solution does not exist/is non-unique: Increase mixing.
                    nonunique =+ 1
                    self.Ra[i], self.Fw[i], self.mask[i,5]  = findMixing(fac, self.Ra[i], self.Fw[i])
                    #print(str(i/self.n) + ': NUMask')                
                    continue
                # This line indicates that a feasible solution has been found: continue to process.
                self.T[i,:] = scaledIntegratedTransport(self.gp, self.L[i,:], self.a[i], self.b[i], self.c[i], self.d[i], self.Sb_X0[i], self.rs[i], self.Xs[i])
                self.Reg[i,:] = computeRegime(self.T[i,:])
                self.LsT[i,:] = computeTheory(self.gp, self.L[i,:], self.Sb_0[i])
                self.We0[i], self.M0[i] = computeWeM(self.gp, self.Ra[i], self.Fr[i], self.Fw[i], self.Sb_X0[i], self.Sb_0[i], self.Xs)
                #self.HGrad[i] = computeHGrad()
                i += 1 #Proceed to next parameter tuple, reset iterative parameters.
                nonunique = 0
                nonphysical = 0
                print(i/self.n)
            self.mixIncrease = self.RaOri/self.Ra
            print(f'rng Fr = {np.amin(self.Fr)} - {np.amax(self.Fr)} and rng Ra {np.amin(self.Ra)} - {np.amax(self.Ra)} and rng Fw {np.amin(self.Fw)} - {np.amax(self.Fw)}')
        else:
            i = 0
            self.mixIncrease = np.ones_like(self.Ra)
            while i < self.n:
                self.L[i,:], self.a[i], self.b[i], self.c[i], self.d[i], self.b0[i], self.c0[i] = nonDim2ODECoeff(self.gp, self.Ra[i], self.Fr[i], self.Fw[i], self.Sc)
                self.D2[i], self.Exx[i,:], self.Exy[i,:] = computeLocalExtrema(self.a[i]/self.d[i], self.b[i]/self.d[i], self.c[i]/self.d[i], 0.0)
                self.Sb_X0[i], self.mask[i,0] = solveCubic(np.array([self.a[i],self.b0[i],self.c0[i], -self.d[i]]))
                self.Sb_0[i], self.Phi_0[i] = processBC(self.gp, self.a[i], self.b[i], self.c[i], self.d[i], self.Ra[i], self.Fr[i], self.Fw[i], self.Sc, self.Sb_X0[i])
                self.rs[i], self.Xs[i], self.rs0[i], self.Xs0[i], self.rs1[i], self.Xs1[i], self.mask[i,1] = computersXs(self.gp, self.a[i], self.b[i], self.c[i], self.d[i], self.Sb_X0[i], self.Fr[i], self.Ra[i], self.Fw[i])
                self.mask[i,2] = computeNU(self.D2[i], self.Exx[i,:], self.Sb_X0[i], self.rs[i])
                if not any(self.mask[i,0:3]): #Solution exists and is unique. Possibly unstable stratification / negative salt.
                    self.mask[i,3], self.mask[i,4], self.maxUN[i,:], self.Phi_c[i], self.Phi_s[i] = computePhysicalMasksPS(self.gp, self.a[i], self.b[i], self.c[i], self.d[i], self.Ra[i], self.Fr[i], self.Fw[i], self.Sc, self.Sb_X0[i], self.Sb_0,self.rs[i])
                    if not any(self.mask[i,3:5]): # Not unstable stratification or negative salinity.
                        self.T[i,:] = scaledIntegratedTransport(self.gp, self.L[i,:], self.a[i], self.b[i], self.c[i], self.d[i], self.Sb_X0[i], self.rs[i], self.Xs[i])
                        self.Reg[i,:] = computeRegime(self.T[i,:])
                        self.LsT[i,:] = computeTheory(self.gp, self.L[i,:], self.Sb_0[i])
                        self.We0[i], self.M0[i] = computeWeM(self.gp, self.Ra[i], self.Fr[i], self.Fw[i], self.Sb_X0[i], self.Sb_0[i], self.Xs)
                    else:
                        self.mask[i,6] = 1
                        self.Reg[i,:] = np.array([.7,.7,.7])
                else:
                    self.mask[i,5] = 1
                    self.Reg[i,:] = np.array([.7,.7,.7])
                self.mask[i,7] = any(self.mask[i,:])
                if np.fmod(10*i, self.n) < .01:
                    print(i/self.n)
                i += 1 #Proceed to next parameter tuple, reset iterative parameters.
            self.maskOri = self.mask

        print('Finished Experiment')

        return