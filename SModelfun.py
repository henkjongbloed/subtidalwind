import numpy as np
import matplotlib as mp
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors as co
#import matplotlib.colors as colors
from scipy.interpolate import griddata
from scipy.integrate import trapz,cumtrapz
from matplotlib.animation import FuncAnimation
from GeneralModelFun import computeNU,computePhysicalMasks, scaledIntegratedTransport, processBC, nonDim2ODECoeff, computeU, computeS, scaledTransport, solveODE, computersXs, solveCubic, computeD3, computeLocalExtrema, computeRegime, getGrid


class SingleNonDim:
    def __init__(self, gp, nondimDict):
        self.gp = gp
        self.gp['Class'] = 'SM'
        #self.name = 'SM'
        self.Ra = nondimDict['Ra']
        self.Fr = nondimDict['Fr']
        self.Fw = nondimDict['Fw']
        self.Sc = nondimDict['Sc']
        return
    
    #def __repr__(self):
    def run(self):
        self.initSM()
        if self.mask[0]:
            print(f'Non-unique mouth salinity, stopped.')
            return self
        else:
            self.solveSM()
            self.processSM()
        return self
    
                #print(f'Unique sal. gradient of {self.Sb_X0}, but illegal salt intrusion length Xs = {self.Xs}')
                #print(f'D3mouth = {self.D3mouth}, and D2 = {self.D2}')
                
    def initSM(self):
        self.L, self.a, self.b, self.c, self.d, self.b0, self.c0 = nonDim2ODECoeff(self.gp, self.Ra, self.Fr, self.Fw, self.Sc)
        #self.D3mouth = computeD3(self.a, self.b0, self.c0, -self.d)
        self.D2, self.Exx, self.Exy = computeLocalExtrema(self.a/self.d, self.b/self.d, self.c/self.d, 0) #Exx = [xright, xleft, xsaddle]
        self.mask = np.empty(5, dtype = bool)
        self.Sb_X0, self.mask[0] = solveCubic(np.array([self.a,self.b0,self.c0, -self.d]))
        return   

    def solveSM(self):
        self.Sb_0, self.Phi_0 = processBC(self.gp, self.Sb_X0, self.a, self.b, self.c, self.d, self.Ra, self.Fr, self.Fw, self.Sc)
        self.rs, self.Xs, self.mask[1] = computersXs(self.gp, self.a, self.b, self.c, self.d, self.Sb_X0)
        if self.mask[1]:
            print('Illegal salt intrusion length')
            return
        self.mask[2] = computeNU(self.gp, self.D2, self.Exx, self.a,self.b,self.c,self.d, self.Sb_X0, self.Sb_0)
        self.r, self.X, self.Sb, self.Sb_X, self.Sb_XX = solveODE(self.gp, self.a, self.b, self.c, self.d, self.Sb_X0, self.rs)
        return
        
    def processSM(self):
        self.Xp, self.sigmap, self.sigma, P = getGrid(self.gp, self.X)
        self.P = P #Shape functions.
        self.TX = scaledTransport(self.L, self.Sb, self.Sb_X)
        self.T = scaledIntegratedTransport(self.gp, self.L, self.a, self.b, self.c, self.Sb_X0, self.rs, self.Xs)
        self.Reg = computeRegime(self.T)
        self.U, self.W, self.Ubar, self.UR, self.UG, self.UW = computeU(self.gp, self.Ra, self.Fr, self.Fw, self.P, self.Sb_X, self.Sb_XX)
        self.S, self.S_X, self.Sbar, self.Sacc, self.Sbar_X, self.Sacc_X = computeS(self.gp, self.Ra, self.Fr, self.Fw, self.Sc, self.P, self.Sb, self.Sb_X, self.Sb_XX)
        self.mask[3], self.mask[4] = computePhysicalMasks(self.gp, self.Ra, self.Fr, self.Fw, self.Sc, self.Sb, self.Sb_X)
        return