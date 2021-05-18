import numpy as np
import numpy.ma as ma
import matplotlib as mp
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors as co
#import matplotlib.colors as colors
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from GeneralModelFun import computePhysicalMasksPS, computeLsTheory, scaledIntegratedTransport, computeNU, computersXs, processBC, computeLocalExtremaVect, nonDim2ODECoeff, solveCubic, Int2, Int3, computeScore, computeRegime, computeMask
from matplotlib import ticker

class ParameterSweep:
    def __init__(self, gp, nondimDict, dim):
        self.gp = gp
        self.nondimDict = nondimDict # Governing parameters
        self.dim = dim               # Dimensional or non-dimensional
        self.nps = nondimDict['nps']
        self.gp['Class'] = 'PS'
        self.Ra = nondimDict['Ra']
        self.Fr = nondimDict['Fr']
        self.Fw = nondimDict['Fw']
        self.Sc = nondimDict['Sc']
        self.n = nondimDict['n']
        return
    
    def run(self):
        self.initPS()                  # Contstruct b and c, b0, b0, b00, c00
        self.solvePS()                 # Compute all relevant quantities (masks, )
        return self

    def initPS(self):
        self.L, self.a, self.b, self.c, self.d, self.b0, self.c0 = nonDim2ODECoeff(self.gp, self.Ra, self.Fr, self.Fw, self.Sc)
        self.D2, self.Exx, self.Exy = computeLocalExtremaVect(self.a/self.d, self.b/self.d, self.c/self.d, np.zeros(self.n)) #Exx = [xright, xleft, xsaddle]
        self.mask = np.empty(shape = (self.n,6), dtype = bool)
        self.Sb_X0 = np.empty(self.n)
        for i in range(self.n):
            self.Sb_X0[i], self.mask[i,0] = solveCubic(np.array([self.a[i],self.b0[i],self.c0[i], -self.d[i]]))
        self.Sb_0, self.Phi_0 = processBC(self.gp, self.Sb_X0, self.a, self.b, self.c, self.d, self.Ra, self.Fr, self.Fw, self.Sc)
        return

    def solvePS(self):
        self.Xs, self.rs = 2*[np.empty(self.n)]
        for i in range(self.n):  
            if self.mask[i,0] == True: #Solution is non unique -> do nothing
                self.Xs[i], self.rs[i], self.mask[i,1] = 0, 0, True
            else:
                self.rs[i], self.Xs[i], self.mask[i,1] = computersXs(self.gp, self.a[i], self.b[i], self.c[i], self.d[i], self.Sb_X0[i])
        
        for i in range(self.n):
            if any(self.mask[i,0:2]):
                self.mask[i,2] = True
            else:
                self.mask[i,2] = computeNU(self.gp, self.D2[i], self.Exx[i,:], self.a[i], self.b[i], self.c[i], self.d[i], self.Sb_X0[i], self.Sb_0[i])
        self.T, self.Reg, self.LsT = np.empty(shape = (self.n, 8)), np.empty(shape = (self.n, 3)), np.empty(shape = (self.n, 7))
        for i in range(self.n):
            if any(self.mask[i,0:2]):
                pass
            else:
                self.T[i,:] = scaledIntegratedTransport(self.gp, self.L[i,:], self.a[i], self.b[i], self.c[i], self.Sb_X0[i], self.rs[i], self.Xs[i])
                self.Reg[i,:] = computeRegime(self.T[i,:])
                self.LsT[i,:] = computeLsTheory(self.gp, self.L[i,:], self.Sb_0[i])
                self.mask[i,3], self.mask[i,4] = computePhysicalMasksPS(self.gp, self.a[i], self.b[i], self.c[i], self.d[i], self.Ra[i], self.Fr[i], self.Fw[i], self.Sc, self.Sb_X0[i], self.rs[i])
        
        self.mask[:,5] = np.count_nonzero(self.mask[:,1:5], axis = 1)
        #self.aTT = [it.reshape(self.nps) for it in aTT]
        #self.Xa, self.ra, self.Ls = Xa.reshape(self.nps), ra.reshape(self.nps), Ls.reshape(self.nps)
        #self.Sb_X0, self.Phi_0, self.Sb_0 = Sb_X0.reshape(self.nps), Phi_0.reshape(self.nps), self.Sb_0.reshape(self.nps)
        #self.Reg = Reg
        #if self.dim == 0:
        #    self.Ra, self.Fr, self.Fw = self.Ra.reshape(self.nps), self.Fr.reshape(self.nps), self.Fw.reshape(self.nps)
        return