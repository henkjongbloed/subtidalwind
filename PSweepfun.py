import numpy as np
import numpy.ma as ma
import matplotlib as mp
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors as co
#import matplotlib.colors as colors
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from GeneralModelFun import findMixing, computePhysicalMasksPS, computeLsTheory, scaledIntegratedTransport, computeNU, computersXs, processBC, computeLocalExtrema, nonDim2ODECoeff, solveCubic, computeRegime
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
        return
    
    def run(self):
        self.initPS()                  # Contstruct b and c, b0, b0, b00, c00
        self.solvePS()                 # Compute all relevant quantities (masks, )
        return self

    def initPS(self):
        self.a, self.b, self.c, self.d, self.b0, self.c0, self.Sb_X0, self.Phi_0, self.Sb_0, self.Xs, self.rs, self.D2 = np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n), np.empty(self.n)
        self.Exx, self.Exy = np.empty(shape = (self.n, 3)), np.empty(shape = (self.n, 3))
        self.L, self.T, self.Reg, self.LsT = np.empty(shape = (self.n, 8)), np.empty(shape = (self.n, 8)), np.empty(shape = (self.n, 3)), np.empty(shape = (self.n, 7))
        self.mask, self.maskFix = np.full((self.n,7), False, dtype = bool), np.full((self.n,7), False, dtype = bool)
        return
    
    def solvePS(self):
        i = 0
        fac = 2.0
        #incr = False
        #mixPar = initMixPar(gp)
        while i < self.n:
            #j = 0
            self.L[i,:], self.a[i], self.b[i], self.c[i], self.d[i], self.b0[i], self.c0[i] = nonDim2ODECoeff(self.gp, self.Ra[i], self.Fr[i], self.Fw[i], self.Sc)
            self.D2[i], self.Exx[i,:], self.Exy[i,:] = computeLocalExtrema(self.a[i]/self.d[i], self.b[i]/self.d[i], self.c[i]/self.d[i], 0.0)
            self.Sb_X0[i], self.mask[i,0] = solveCubic(np.array([self.a[i],self.b0[i],self.c0[i], -self.d[i]]))
            self.Sb_0[i], self.Phi_0[i] = processBC(self.gp, self.a[i], self.b[i], self.c[i], self.d[i], self.Ra[i], self.Fr[i], self.Fw[i], self.Sc, self.Sb_X0[i])
            self.rs[i], self.Xs[i], self.mask[i,1] = computersXs(self.gp, self.a[i], self.b[i], self.c[i], self.d[i], self.Sb_X0[i])
            self.mask[i,2] = computeNU(self.gp, self.D2[i], self.Exx[i,:], self.a[i], self.b[i], self.c[i], self.d[i], self.Sb_X0[i], self.Sb_0[i])
            if not any(self.mask[i,0:3]): #Solution exists and is unique. Possibly unstable stratification / negative salt.
                self.T[i,:] = scaledIntegratedTransport(self.gp, self.L[i,:], self.a[i], self.b[i], self.c[i], self.Sb_X0[i], self.rs[i], self.Xs[i])
                self.Reg[i,:] = computeRegime(self.T[i,:])
                self.LsT[i,:] = computeLsTheory(self.gp, self.L[i,:], self.Sb_0[i])
                self.mask[i,3], self.mask[i,4] = computePhysicalMasksPS(self.gp, self.a[i], self.b[i], self.c[i], self.d[i], self.Ra[i], self.Fr[i], self.Fw[i], self.Sc, self.Sb_X0[i], self.rs[i])
                #if incr = True:
                #    fac = 
                if any(self.mask[i,3:4]):
                    self.Ra[i], self.Fw[i], self.mask[i,6] = findMixing(fac, self.Ra[i], self.Fw[i])
                    continue
            else: #Solution does not exist/is non-unique: Increase mixing.
                self.Ra[i], self.Fw[i], self.mask[i,5]  = findMixing(fac, self.Ra[i], self.Fw[i])
                #incr = True
                continue
            i += 1 #Proceed to next parameter tuple, reset iterative parameters.
            
        #self.mask[:,6] = np.count_nonzero(self.mask[:,0:6], axis = 1)
        return