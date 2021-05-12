import numpy as np
import matplotlib as mp
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors as co
#import matplotlib.colors as colors
from scipy.interpolate import griddata
from scipy.integrate import trapz,cumtrapz
from matplotlib.animation import FuncAnimation
from GeneralModelFun import solveCubic, computeD2, computeD3, addC, formFunctions, computeLocalExtrema, Int3, Int2, scoreIJ, computeRegime


class SingleNonDim: #to do: Make dimensional model
    def __init__(self, gp, nondimDict):
        self.gp = gp
        self.Ra = nondimDict['Ra']
        self.Fr = nondimDict['Fr']
        self.Fw = nondimDict['Fw']
        self.Sc = nondimDict['Sc']
        #self.Ft = (0.035*0.028*0.0026*self.Ra)**(-1/2)
        return
    
    #def __repr__(self):
    def run(self):
        self.addODEC()
        self.mouthMC()
        self.D3mouth = computeD3(self.a, self.b0, self.c0, -self.d)
        self.D2, self.Exx, self.Exy = computeLocalExtrema(self.a/self.d, self.b/self.d, self.c/self.d, 0) #Exx = [xright, xleft, xsaddle]
        if self.Sb_X0 != -1:
            self.processMouth()
            self.computeXa()
            if self.Xa < 0:
                self.solveODE()
                self.processMC()
                self.computeTransport()
                self.computeScore()
            else:
                print(f'Unique mouth salinity of {self.Sb_X0}, but illegal salt intrusion length (ra, X_a) = ({self.ra}, {self.Xa})')
                print(f'D3mouth = {self.D3mouth}, and D2 = {self.D2}')
        else:
            print(f'Non-unique mouth salinity, stopped. D3mouth = {self.D3mouth}, and D2 = {self.D2}')
        return self
    
    def addODEC(self):
        C = self.gp['C']
        Ra, Fr, Fw, Sc = self.Ra, self.Fr, self.Fw, self.Sc
        L = [C[0]*Sc*Ra**3,                     #GG cubed
            C[1]*Sc*Ra**2*Fr,                   #GR squared
            C[2]*Sc*Ra**2*Fw,                   #GW squared
            C[3]*Sc*Ra*Fr**2,                   #RR
            C[4]*Sc*Ra*Fr*Fw,                   #RW
            C[5]*Sc*Ra*Fw**2,                   #WW
            1.0,                                #D
            Fr]                                 #F
        
        self.a = L[0]
        self.b = L[1] + L[2]
        self.c = L[3] + L[4] + L[5] + L[6]
        self.d = L[7]
        
        self.b0 = self.b + self.d*Sc*Ra**2*C[10]
        self.c0 = self.c + self.d*Sc*Ra*(Fr*C[9] + Fw*C[11])
        
        self.L = L
        return   
    
    def mouthMC(self):
        '''Computes single Sb_X0 based on parameter values'''
        p = np.array([self.a,self.b0,self.c0, -self.d]) #Coefficients of cubic equation at X=0
        self.Sb_X0, _ = solveCubic(p)
        return
    
    def solveODE(self):
        C = self.gp['C']
        ns = self.gp['SMGrid'][0]
        a, b, c, d, Sb_X0, ra = self.a, self.b, self.c, self.d, self.Sb_X0, self.ra
        r = np.linspace(ra, 0, ns)
        X = (3/2*a*Sb_X0**2*(np.exp(2*r)-1) + 2*b*Sb_X0*(np.exp(r)-1) + c*r)/d
        Sb = (a*Sb_X0**3*np.exp(3*r) + b*Sb_X0**2*np.exp(2*r) + c*Sb_X0*np.exp(r))/d
        Sb_X = Sb_X0*np.exp(r)
        Sb_XX = Sb_X/(3*a*Sb_X**2 + 2*b)
        self.X, self.Sb, self.Sb_X, self.Sb_XX, self.r, self.ra = X, Sb, Sb_X, Sb_XX, r, ra
        return
    
    def processMouth(self):
        C = self.gp['C']
        Sb_X0, a, b, c, d = self.Sb_X0, self.a, self.b, self.c, self.d
        Sb_0 = (a*Sb_X0**3 + b*Sb_X0**2 + c*Sb_X0) / d
        S_00 = 1 - self.Sc*self.Ra*Sb_X0*(self.Fr*C[6] + self.Ra*Sb_X0*C[7] + self.Fw*C[8])
        Phi_0 = 1 - S_00
        return
    
    def computeXa(self):
        Sb_X0, alpha, a, b, c, d = self.Sb_X0, self.gp['alpha'], self.a, self.b, self.c, self.d
        if Sb_X0 == -1: #Solution is non unique
            self.Xa = -1
            return
        expr, _ = solveCubic(np.array([a*Sb_X0**2, b*Sb_X0, c, -alpha*(a*Sb_X0**2 + b*Sb_X0 + c)])) #rt = exp(r)
        if expr < 1.0:
            ra = np.log(expr)
            self.Xa = (3/2*a*Sb_X0**2*(np.exp(2*ra)-1) + 2*b*Sb_X0*(np.exp(ra)-1) + c*ra)/d
        else:
            self.Xa = -1
            return
        #print(ratemp)
        self.ra = ra
        return
    
    def computeTransport(self):
        #C = self.gp['C']
        #Ra, Fr, Fw, Sc = self.Ra, self.Fr, self.Fw, self.Sc
        Sb, Sb_X, X = self.Sb, self.Sb_X, self.X
        alpha = self.gp['alpha']
        bl, bu = self.Sb_X0*np.exp(self.ra), self.Sb_X0
        a, b, c = self.a, self.b, self.c
        Ls = np.abs(self.Xa)
        L = self.L
        
        P = [Sb_X**3/Sb,    #GG
            Sb_X**2/Sb,     #GR
            Sb_X**2/Sb,     #GW
            Sb_X/Sb,#RR
            Sb_X/Sb,#RW
            Sb_X/Sb,#WW
            Sb_X/Sb,#D
            Sb/Sb#F
            ] #For numerical integration.
        
        aT = [Int3(bl, bu, a, b, c), #Analytically integrated P
            Int2(bl, bu, a, b, c),
            Int2(bl, bu, a, b, c),
            -np.log(alpha), #L[0] is just a dummy variable, no length
            -np.log(alpha),
            -np.log(alpha),
            -np.log(alpha),
            Ls
            ]
        
        T = [l*p/L[7] for l,p in zip(L, P)] #transports as function of X
        #nTT = np.array([np.trapz(Ti, X, axis = 0)/Ls for Ti in T]) #X-integrated, scaled transport: Numerical
        aTT = np.array([l*at/(Ls*L[7]) for l, at in zip(L, aT)]) #analytical integration, scaled
        #aTT = 

        self.T, self.aTT = T, aTT
        return
    
    def computeScore(self):
        TT = self.aTT
        s = scoreIJ(TT)
        self.Reg = computeRegime(s)
        #print(self.Reg)
        return

    def processMC(self):
        nsigma = self.gp['SMGrid'][1]
        Ra, Fr, Fw, Sc, BC, X = self.Ra, self.Fr, self.Fw, self.Sc, self.gp['BC'], self.X
        Sb, Sb_X, Sb_XX = self.Sb, self.Sb_X, self.Sb_XX
        C = self.gp['C']
        #nsigma = 31
        sigma = np.linspace(-1, 0, nsigma)

        Xp, sigmap = np.meshgrid(X,sigma)

        P1, P2, P3, P4, P5, P6, P7 = formFunctions(sigmap, BC)
        
        Ubar = Fr*np.ones_like(sigmap)    
        UR = Fr*P1
        UG = Ra*P2*np.matlib.repmat(np.transpose(Sb_X),nsigma,1)
        UW = Fw*P3   
        #temp = np.transpose(Sb)# np.matlib.repmat(np.transpose(Sb),nsigma,1) -> Start here!!!!
        #print(temp)
        self.Sbar = np.matlib.repmat(np.transpose(Sb),nsigma,1) 
        self.Sacc = Sc*Ra*(Fr*P4 + Fw*P6)*np.matlib.repmat(np.transpose(Sb_X),nsigma,1) + Sc*Ra**2*P5*np.matlib.repmat(np.transpose(Sb_X),nsigma,1)**2
        self.S = self.Sbar + self.Sacc #S[0,0] is landward surface, S[-1,-1] is seaward bottom
        self.U =  Ubar + UR + UG + UW
        self.Ubar, self.UR, self.UG, self.UW = Ubar, UR, UG, UW
        Re = 3000 #random
        W = 0*P7 # Revisit this.
        self.W = W/np.mean(abs(W))*np.mean(UG)
        self.sigma, self.Xp, self.sigmap = sigma, Xp, sigmap
        self.maskNU = np.any(Sb[1:] <= Sb[:-1]) # 1 if there are local extrema (if there is non-monotonicity)
        self.maskIS = np.any((self.S[0,:] < self.S[-1,:])) # 1 if there exists a point of unstable stratification
        self.maskNEG = np.any((self.S[-1,:] < 0)) # 1 if there is negative salt
        return