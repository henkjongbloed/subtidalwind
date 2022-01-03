import numpy as np
import matplotlib as mp
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors as co
#import matplotlib.colors as colors
from scipy.interpolate import griddata
from scipy.integrate import trapz,cumtrapz
from matplotlib.animation import FuncAnimation
from generalFun import *


class SingleNonDim:
    def __init__(self, gp, nondimDict):
        self.gp = gp
        self.gp['Class'] = 'SM'
        #self.name = 'SM'
        self.Ra = nondimDict['Ra']
        self.Fr = nondimDict['Fr']
        self.Fw = nondimDict['Fw']
        self.Sc = nondimDict['Sc']
        self.RaOri = self.Ra
        return
    
    def run(self):            
        self.mask = np.empty(5, dtype = bool)
        self.maskOri = np.copy(self.mask)
        if not self.gp['mixAway']:
            self.L, self.a, self.b, self.c, self.d, self.b0, self.c0 = nonDim2ODECoeff(self.gp, self.Ra, self.Fr, self.Fw, self.Sc)
            #self.D3mouth = computeD3(self.a, self.b0, self.c0, -self.d)
            self.D2, self.Exx, self.Exy = computeLocalExtrema(self.a/self.d, self.b/self.d, self.c/self.d, 0) #Exx = [xright, xleft, xsaddle]
            self.Sb_X0, self.mask[0] = solveCubic(np.array([self.a,self.b0,self.c0, -self.d]))
            #print(self.Sb_X0)
            #print(self.Fr/self.L[0]) 
            if self.mask[0]:
                print(f'Non-unique mouth salinity, stopped.')
                return self
            self.Sb_0, self.Phi_0 = processBC(self.gp, self.Sb_X0, self.a, self.b, self.c, self.d, self.Ra, self.Fr, self.Fw, self.Sc)
            self.rs, self.Xs, self.rs0, self.Xs0, self.rs1, self.Xs1, self.mask[1] = computersXs(self.gp, self.a, self.b, self.c, self.d, self.Sb_X0, self.Fr, self.Ra, self.Fw)
            if self.mask[1]:
                print('Illegal salt intrusion length')
                return
            self.mask[2] = computeNU(self.D2, self.Exx, self.Sb_X0, self.rs)
            self.r, self.X, self.Sb, self.Sb_X, self.Sb_XX = solveODE(self.gp, self.a, self.b, self.c, self.d, self.Sb_X0, self.rs)
        else:
            i = 0
            fac = self.gp['m']
            #incr = False
            #mixPar = initMixPar(gp)
            nonunique = 0
            nonphysical = 0
            while i < 1:
                self.L, self.a, self.b, self.c, self.d, self.b0, self.c0 = nonDim2ODECoeff(self.gp, self.Ra, self.Fr, self.Fw, self.Sc)
                self.D2, self.Exx, self.Exy = computeLocalExtrema(self.a/self.d, self.b/self.d, self.c/self.d, 0) #Exx = [xright, xleft, xsaddle]
                self.Sb_X0, self.mask[0] = solveCubic(np.array([self.a,self.b0,self.c0, -self.d]))            
                self.Sb_0, self.Phi_0 = processBC(self.gp, self.Sb_X0, self.a, self.b, self.c, self.d, self.Ra, self.Fr, self.Fw, self.Sc)
                self.rs, self.Xs, self.mask[1] = computersXs(self.gp, self.a, self.b, self.c, self.d, self.Sb_X0)
                self.mask[2] = computeNU(self.D2, self.Exx, self.Sb_X0, self.rs)
                print(self.mask[2])
                print(self.Sb_X0)
                print(self.Fr/self.L[6]) 
                if nonunique == 0: #Save the original mask for plotting later on.
                    self.maskOri[0:3] = self.mask[0:3]
                if not any(self.mask[0:3]):
                    self.T = scaledIntegratedTransport(self.gp, self.L, self.a, self.b, self.c, self.d, self.Sb_X0, self.rs, self.Xs)
                    self.Reg = computeRegime(self.T)
                    self.LsT = computeLsTheory(self.gp, self.L, self.Sb_0)
                    self.mask[3], self.mask[4], maxUN = computePhysicalMasksPS(self.gp, self.a, self.b, self.c, self.d, self.Ra, self.Fr, self.Fw, self.Sc, self.Sb_X0, self.rs)
                    if nonphysical == 0:
                        self.maskOri[3:5] = self.mask[3:5]
                        self.maxUN = maxUN
                    if any(self.mask[3:5]):
                        nonphysical =+ 1
                        self.Ra, self.Fw, self.mask[6] = findMixing(fac, self.Ra, self.Fw)
                        print('Fixed Physical mask')
                        continue
                else: #Solution does not exist/is non-unique: Increase mixing.
                    nonunique =+ 1
                    self.Ra, self.Fw, self.mask[5]  = findMixing(fac, self.Ra, self.Fw)
                    print('Fixed Math mask')
                    continue
                i += 1 #Proceed to next parameter tuple, reset iterative parameters.
                print(i)
                nonunique = 0
                nonphysical = 0
            self.mixIncrease = self.RaOri/self.Ra
            print(self.mixIncrease)
        self.r, self.X, self.Sb, self.Sb_X, self.Sb_XX = solveODE(self.gp, self.a, self.b, self.c, self.d, self.Sb_X0, self.rs)
        self.processSM()
        return self
        
    def processSM(self):
        self.Xp, self.sigmap, self.sigma, P = getGrid(self.gp, self.X)
        self.P = P #Shape functions.
        self.TX = scaledTransport(self.gp,self.L, self.Sb, self.Sb_X)
        self.T = scaledIntegratedTransport(self.gp, self.L, self.a, self.b, self.c, self.d,self.Sb_X0, self.rs, self.Xs)
        self.Reg = computeRegime(self.T)
        self.U, self.W, self.Ubar, self.UR, self.UG, self.UW = computeU(self.gp, self.Ra, self.Fr, self.Fw, self.P, self.Sb_X, self.Sb_XX)
        self.S, self.S_X, self.Sbar, self.Sacc, self.Sbar_X, self.Sacc_X = computeS(self.gp, self.Ra, self.Fr, self.Fw, self.Sc, self.P, self.Sb, self.Sb_X, self.Sb_XX)
        self.mask[3], self.mask[4] = computePhysicalMasks(self.gp, self.Ra, self.Fr, self.Fw, self.Sc, self.Sb, self.Sb_X)
        return