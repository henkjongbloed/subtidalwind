import numpy as np
import matplotlib as mp
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors as co
#import matplotlib.colors as colors
from scipy.interpolate import griddata
from scipy.integrate import trapz,cumtrapz
from matplotlib.animation import FuncAnimation
from GeneralModelFun import solveCubic, computeD2, computeD3, dim2nondim, addC, formFunctions, computeLocalExtrema, I0,I1,I2,I3, scoreIJ, computeRegime


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
        self.D3mouth = computeD3(1.0, self.b0, self.c0, -1.0)
        self.D2, self.Exx, self.Exy = computeLocalExtrema(1, self.b, self.c, 0) #Exx = [xright, xleft, xsaddle]
        if self.Sb_X0 != -1:
            self.processMouth()
            self.computeXa()
            if self.ra > 0:
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
        self.b = Sc**(1/3)*(C[7]*Fr**(2/3) + C[8]*Fr**(-1/3)*Fw)
        self.c = Sc**(2/3)*(C[9]*Fr**(4/3) + C[10]*Fr**(1/3)*Fw + C[11]*Fr**(-2/3)*Fw**2) + C[12]*Sc**(-1/3)*Fr**(-2/3)*Ra**(-1)
        self.b0 = self.b + C[13]*Sc**(1/3)*Fr**(2/3)*C[18]
        self.c0 = self.c + C[12]*Sc**(2/3)*Fr**(1/3)*(C[17]*Fr + C[19]*Fw)
        self.b00 = C[13]*Sc**(1/3)*Fr**(2/3)*C[15]
        self.c00 = C[12]*Sc**(2/3)*Fr**(1/3)*(C[14]*Fr +C[16]*Fw)
        return   
    
    def mouthMC(self):
        '''Computes single Sb_X0 based on parameter values'''
        p = np.array([1,self.b0,self.c0,-1]) #Coefficients of cubic equation at X=0
        self.Sb_X0, _ = solveCubic(p)
        return
    
    def solveODE(self):
        ns = self.gp['SMGrid'][0]
        b, c, Sb_X0, ra = self.b, self.c, self.Sb_X0, self.ra
        r = np.linspace(ra, 0, ns)
        #if self.D2 < 0:
        #    r = np.linspace(ra, 0, ns) #ra should be positive, so this is a sequence with 'negative increments'
        #else:
        #    r = np.linspace(ra, 0, ns) #ra should be positive, so this is a sequence with 'negative increments'
        #    pass
        X = 3/2*Sb_X0**2*(np.exp(-2*r)-1) + 2*b*Sb_X0*(np.exp(-r)-1) - c*r
        Sb = Sb_X0**3*np.exp(-3*r) + b*Sb_X0**2*np.exp(-2*r) + c*Sb_X0*np.exp(-r)
        Sb_X = Sb_X0*np.exp(-r)
        Sb_XX = Sb_X/(3*Sb_X**2 + 2*b)
        self.X, self.Sb, self.Sb_X, self.Sb_XX, self.r, self.ra = X, Sb, Sb_X, Sb_XX, r, ra
        return
    
    def processMouth(self):
        Sb_X0, b, c, b00, c00 = self.Sb_X0, self.b, self.c, self.b00, self.c00
        self.Sb_0 = Sb_X0**3 + b*Sb_X0**2 + c*Sb_X0
        self.S_00 = self.Sb_0 + b00*Sb_X0**2 + c00*Sb_X0
        self.Phi_0 = 1 - self.S_00
        return
    
    def computeXa(self):
        Sb_X0, alpha, b, c = self.Sb_X0, self.gp['alpha'], self.b, self.c
        if Sb_X0 == -1: #Solution is non unique
            self.Xa = -1
            return
        p = np.array([Sb_X0**2, b*Sb_X0, c, -alpha*(Sb_X0**2 + b*Sb_X0 + c)])
        ratemp, _ = solveCubic(p)
        #print(ratemp)
        ra = -np.log(ratemp)
        self.Xa = 3/2*Sb_X0**2*(np.exp(-2*ra)-1) + 2*b*Sb_X0*(np.exp(-ra)-1) - c*ra
        self.ra = ra
        return
    
    def computeTransport(self):
        C = self.gp['C']
        Ra, Fr, Fw, Sc = self.Ra, self.Fr, self.Fw, self.Sc
        Sb, Sb_X, X = self.Sb, self.Sb_X, self.X
        alpha = self.gp['alpha']
        bl, bu = self.Sb_X0*np.exp(-self.ra), self.Sb_X0
        b, c = self.b, self.c
        Ls = np.abs(self.Xa)
        L = np.array([1.0, #GG
            C[7]*Sc**(1/3)*Fr**(2/3),#GR
            C[8]*Sc**(1/3)*Fr**(-1/3)*Fw,#GW
            C[9]*Sc**(2/3)*Fr**(4/3),#RR
            C[10]*Sc**(2/3)*Fr**(1/3)*Fw,#RW
            C[11]*Sc**(2/3)*Fr**(-2/3)*Fw**2,#WW
            C[12]*Sc**(-1/3)*Fr**(-2/3)*Ra**-1,#D
            -1.0])#F
        
        P = [Sb_X**3/Sb,    #GG
            Sb_X**2/Sb,     #GR
            Sb_X**2/Sb,     #GW
            Sb_X/Sb,#RR
            Sb_X/Sb,#RW
            Sb_X/Sb,#WW
            Sb_X/Sb,#D
            Sb/Sb#F
            ] #For numerical integration.
        
        aT = [3*(I3(bu,1,b,c)-I3(bl,1,b,c)) + 2*b*(I2(bu,1,b,c)-I2(bl,1,b,c)) + c*(I1(bu,1,b,c)-I1(bl,1,b,c)), #Analytically integrated P
            3*(I2(bu,1,b,c)-I2(bl,1,b,c)) + 2*b*(I1(bu,1,b,c)-I1(bl,1,b,c)) + c*(I0(bu,1,b,c)-I0(bl,1,b,c)),
            3*(I2(bu,1,b,c)-I2(bl,1,b,c)) + 2*b*(I1(bu,1,b,c)-I1(bl,1,b,c)) + c*(I0(bu,1,b,c)-I0(bl,1,b,c)),
            np.log(1/alpha),
            np.log(1/alpha),
            np.log(1/alpha),
            np.log(1/alpha),
            np.abs(self.Xa)
            ]
        
        T = [l*p for l,p in zip(L,P)] #transports as function of X
        #nTT = np.array([np.trapz(Ti, X, axis = 0)/Ls for Ti in T]) #X-integrated, scaled transport: Numerical
        aTT = [l*at/Ls for l,at in zip(L,aT)] #analytical integration, scaled
        aTT = np.array(aTT)
        

        #print(nTT)
        #print(aTT)
        self.T,  self.aTT= T,  aTT
        return
    
    def computeScore(self):
        TT = self.aTT
        s = scoreIJ(TT)
        self.Reg = computeRegime(s)
        #print(self.Reg)
        return

    def processMC(self):
        nsigma = self.gp['SMGrid'][1]
        Fr, Fw, Sc, BC, X = self.Fr, self.Fw, self.Sc, self.gp['BC'], self.X
        Sb, Sb_X, Sb_XX = self.Sb, self.Sb_X, self.Sb_XX
        C = self.gp['C']
        #nsigma = 31
        sigma = np.linspace(-1, 0, nsigma)

        Xp, sigmap = np.meshgrid(X,sigma)

        P1, P2, P3, P4, P5, P6, P7 = formFunctions(sigmap, BC)
        
        Ubar = Fr*np.ones_like(sigmap)    
        UR = Fr*P1
        UG = C[12]*Sc**(-1/3)*Fr**(1/3)*P2*np.matlib.repmat(np.transpose(Sb_X),nsigma,1)
        UW = Fw*P3   
        #temp = np.transpose(Sb)# np.matlib.repmat(np.transpose(Sb),nsigma,1) -> Start here!!!!
        #print(temp)
        self.Sbar = np.matlib.repmat(np.transpose(Sb),nsigma,1) 
        self.Sacc = C[12]*Sc**(2/3)*Fr**(1/3)*(Fr*P4 + Fw*P6)*np.matlib.repmat(np.transpose(Sb_X),nsigma,1) + C[13]*Sc**(1/3)*Fr**(2/3)*P5*np.matlib.repmat(np.transpose(Sb_X),nsigma,1)**2
        self.S = self.Sbar + self.Sacc #S[0,0] is landward surface, S[-1,-1] is seaward bottom
        self.U =  Ubar + UR + UG + UW
        self.Ubar, self.UR, self.UG, self.UW = Ubar, UR, UG, UW
        Re = 3000 #random
        W = -C[13]*Fr**(2/3)*Sc**(-2/3)*Re**(-1)*np.matlib.repmat(np.transpose(Sb_XX),nsigma,1)*P7
        self.W = W/np.mean(abs(W))*np.mean(UG)
        self.sigma, self.Xp, self.sigmap = sigma, Xp, sigmap
        self.maskNU = np.any(Sb[1:] <= Sb[:-1]) # 1 if there are local extrema (if there is non-monotonicity)
        self.maskIS = np.any((self.S[0,:] < self.S[-1,:])) # 1 if there exists a point of unstable stratification
        self.maskNEG = np.any((self.S[-1,:] < 0)) # 1 if there is negative salt
        return
    '''
    #def T2Score():
    maskNU, maskIS, maskNEG = maskPS(b,c, Ra, Fr, Fw, Sc, Sb_X0, ra, ns, C)
    
    def maskSM(self):
        Sb = Sb_X0**3*np.exp(-3*r) + b*Sb_X0**2*np.exp(-2*r) + c*Sb_X0*np.exp(-r)
        Sb_X = Sb_X0*np.exp(-r)

        SXsurface = Sb + C[12]*Sc**(2/3)*Fr**(1/3)*(Fr*P40 + Fw*P60)*Sb_X + C[13]*Sc**(1/3)*Fr**(2/3)*P50*Sb_X**2
        SXbottom = Sb + C[12]*Sc**(2/3)*Fr**(1/3)*(Fr*P4 + Fw*P6)*Sb_X + C[13]*Sc**(1/3)*Fr**(2/3)*P5*Sb_X**2
        maskNU = np.any(Sb[1:] <= Sb[:-1]) # 1 if there are local extrema (if there is non-monotonicity)
        maskIS = np.any((SXbottom < SXsurface)) # 1 if there exists a point of unstable stratification
        maskNEG = np.any((SXsurface < 0)) # 1 if there is negative salt
        return bool(maskNU), bool(maskIS), bool(maskNEG)        
    '''
    #def dimModel(self):
        #'''Convert dimensionless model back to dimensional model'''
        
    '''
    def plotSalt(self):
        X, Xp, sigma, sigmap = self.X, self.Xp, self.sigma, self.sigmap
        S, Sb, Sacc = self.S, self.Sb, self.Sacc
        T = self.T
        S = np.ma.masked_where(S<0, S)
        labT = ['GC-GC', 'GC-Riv', 'GC-Wind', 'Riv-Riv', 'Riv-Wind', 'Wind-Wind', 'Mixing']
        cmap = plt.cm.get_cmap("jet")
        _, axs = plt.subplots(2,2)
        plt.tight_layout()
        f1 = axs[0,0].contourf(Xp, sigmap, S, 50, cmap=cmap, corner_mask = True)
        axs[0,0].title.set_text('Dimensionless salinity')
        axs[0,0].set_xlabel(r'$X$')
        axs[0,0].set_ylabel(r'$\sigma$')
        plt.colorbar(f1, ax=axs[0,0])

        axs[0,1].plot(X, Sb, label = 'Parametric')
        axs[0,1].plot(self.Xe, self.Sbe, label = 'Euler')
        axs[0,1].title.set_text('Dimensionless z-averaged salinity')
        axs[0,1].set_xlabel(r'$X$')
        axs[0,1].set_ylabel(r'$\bar{\Sigma}_X$')
        axs[0,1].legend()
        axs[0,1].grid(True)
        
        f3 = axs[1,0].contourf(Xp, sigmap, Sacc, 50, cmap=cmap, corner_mask = True)
        axs[1,0].title.set_text('Dimensionless varying salinity')
        axs[1,0].set_xlabel(r'$X$')
        axs[1,0].set_ylabel(r'$\sigma$')
        plt.colorbar(f3, ax=axs[1,0])
        
        
        for t in range(len(T)):
            axs[1,1].plot(X, T[t], label = labT[t])
        axs[1,1].title.set_text('Transport Contributions')
        axs[1,1].set_xlabel(r'$X$')
        axs[1,1].set_ylabel('Relative Contribution')
        axs[1,1].legend()
        axs[1,1].grid(True)    
    
        
        axs[1,1].plot(S[:,-1], sigma, 0*S[:,-1] + Sb[-1], sigma)
        axs[1,1].title.set_text('Salinity at X = 0')
        axs[1,1].set_xlabel(r'$\Sigma(0,\sigma)$')
        axs[1,1].set_ylabel(r'$\sigma$')
        axs[1,1].grid(True)
        
        
    def plotOverview(self):
        X, Xp, sigmap, sigma = self.X, self.Xp, self.sigmap, self.sigma
        S, Sb = self.S, self.Sb
        
        Sb_X, r = self.Sb_X, self.r
        
        Sbe, Sb_Xe, Xe = self.Sbe, self.Sb_Xe, self.Xe
        
        U, W = self.U, self.W
        T = self.T

        labT = ['GG', 'GR', 'GW', 'RR', 'RW', 'WW', 'Di', 'Fl']
        cmap = 'Blues'
        
        xnew = np.linspace(np.min(X), 0, np.max(np.shape(X)))
        Xp_interp, _ = np.meshgrid(xnew, sigma)
        
        U_interp  = griddata((np.ravel(sigmap), np.ravel(Xp)), np.ravel(U) , (sigmap, Xp_interp), method='linear')
        W_interp  = griddata((np.ravel(sigmap), np.ravel(Xp)), np.ravel(W) , (sigmap, Xp_interp), method='linear')

        
        _, axs = plt.subplots(3,3)
        plt.tight_layout()
        
        f1 = axs[0,0].contourf(Xp, sigmap, S, 50, cmap=cmap, corner_mask = True)
        axs[0,0].contour(f1, levels = np.linspace(0,1,10), colors = 'k', linewidths = 0.5)
        axs[0,0].contour(f1, levels = np.linspace(0,1,50), colors = 'k', linewidths = 0.5, alpha = 0.2)
        axs[0,0].title.set_text('Dimensionless salinity')
        axs[0,0].set_xlabel(r'$X$')
        axs[0,0].set_ylabel(r'$\sigma$')
        if np.amin(S) < 0:
            axs[0,0].contourf(Xp, sigmap, S, levels = [np.amin(S), 0, np.amax(S)], cmap = cmap, corner_mask = True, hatches=["//", ""], alpha = 0)
            axs[0,0].contour(f1, levels = [0], colors='w', linewidths = 1.5)
        plt.colorbar(f1, ax=axs[0,0])
        
        for t in range(len(T)):
            axs[1,0].plot(X, T[t], label = labT[t]+': '+np.array2string(*self.aTT[t]))
        axs[1,0].title.set_text('Transport Contributions')
        axs[1,0].set_xlabel(r'$X$')
        axs[1,0].set_ylabel('Relative Contribution')
        axs[1,0].legend()
        axs[1,0].grid(True)
        
        mag = np.sqrt(U_interp**2 + W_interp**2)
        f2 = axs[2,0].contourf(Xp, sigmap, U, 50, cmap = cmap, corner_mask = True)
        axs[2,0].streamplot(Xp_interp, sigmap, U_interp, W_interp, density=1, color='k', linewidth=3*mag/mag.max())
        axs[2,0].title.set_text('Dimensionless Flow')
        axs[2,0].set_xlabel(r'$X$')
        axs[2,0].set_ylabel(r'$\sigma$')
        axs[2,0].contour(f2, levels = [0], colors='w', linewidths = 1.5)
        plt.colorbar(f2, ax=axs[2,0])
        
        sbxmin = min([self.Exx[1], np.min(Sb_X)])
        sbxmax = max([self.Exx[0], np.max(Sb_X)])
    
        Sb_Xplot = np.linspace(sbxmin, sbxmax, 201)
        Sbplot = np.polyval([1,self.b,self.c,0], Sb_Xplot)
        
        
                
        axs[0,1].plot(Sb_Xplot, Sbplot, ls = 'dotted', label = 'Curve')
        axs[0,1].plot(Sb_X, Sb, lw = 2, label = 'Realised')
        axs[0,1].scatter(self.Exx, self.Exy, marker = 'o', label = 'Loc. Extr.')
        axs[0,1].title.set_text(r'Parametric: $\bar{\Sigma}_X - \bar{\Sigma}$')
        axs[0,1].set_xlabel(r'$\bar{\Sigma}_X$')
        axs[0,1].set_ylabel(r'$\bar{\Sigma}$')
        axs[0,1].grid(True)
        axs[0,1].legend()
        
        sbxmin = min([self.Exx[1], np.min(Sb_Xe)])
        sbxmax = max([self.Exx[0], np.max(Sb_Xe)])
    
        Sb_Xplot = np.linspace(sbxmin, sbxmax, 201)
        Sbplot = np.polyval([1,self.b,self.c,0], Sb_Xplot)
        
        axs[0,2].plot(Sb_Xplot, Sbplot, ls = 'dotted', label = 'Curve')
        axs[0,2].plot(Sb_Xe, Sbe, lw = 2, label = 'Realised')
        axs[0,2].scatter(self.Exx, self.Exy, marker = 'o', label = 'Loc. Extr.')
        axs[0,2].title.set_text(r'Euler: $\bar{\Sigma}_X - \bar{\Sigma}$')
        axs[0,2].set_xlabel(r'$\bar{\Sigma}_X$')
        axs[0,2].set_ylabel(r'$\bar{\Sigma}$')
        axs[0,2].grid(True)
        axs[0,2].legend()
        
        axs[1,1].plot(X, Sb, label = 'Parametric')
        axs[1,1].plot(self.Xe, self.Sbe, label = 'Euler')
        axs[1,1].title.set_text(r'$\bar{\Sigma}$: Par. and Euler')
        axs[1,1].set_xlabel('X')
        axs[1,1].set_ylabel(r'$\bar{\Sigma}$')
        axs[1,1].legend()
        axs[1,1].grid(True)
        
        axs[1,2].plot(self.Xe, self.Sbe, label = 'Euler')
        axs[1,2].title.set_text(r'$\bar{\Sigma}$: Euler')
        axs[1,2].set_xlabel('X')
        axs[1,2].set_ylabel(r'$\bar{\Sigma}$')
        axs[1,2].grid(True)
        
        axs[2,1].plot(r, X/np.max(np.abs(X)), label = r'$X$')
        axs[2,1].plot(r, Sb, label = r'$\bar{\Sigma}$')
        axs[2,1].plot(r, Sb_X/np.max(np.abs(Sb_X)), label = r'$\bar{\Sigma}_X$')
        axs[2,1].title.set_text('Parametrisation')
        axs[2,1].set_xlabel(r'$r$')
        #axs[1,2].set_ylabel(r'$\bar{\Sigma}')
        axs[2,1].legend()
        axs[2,1].grid(True)
        
        axs[2,2].plot(r, Xe/np.max(np.abs(Xe)), label = r'$X$')
        axs[2,2].plot(r, Sbe, label = r'$\bar{\Sigma}$')
        axs[2,2].plot(r, Sb_Xe/np.max(np.abs(Sb_Xe)), label = r'$\bar{\Sigma}_X$')
        axs[2,2].title.set_text('Parametrisation')
        axs[2,2].set_xlabel(r'$r$')
        #axs[1,2].set_ylabel(r'$\bar{\Sigma}')
        axs[2,2].legend()
        axs[2,2].grid(True)
    
    def plotFlow(self):
        U, UR, UG, UW, W, Ubar = self.U, self.UR, self.UG, self.UW, self.W, self.Ubar
        X, sigma, Xp, sigmap = self.X, self.sigma, self.Xp, self.sigmap
        nX = np.max(np.shape(X))
        Xm = [int(np.round(nX/4)), int(np.round(nX/2)), int(np.round(nX*3/4))]
        #print(xm)
        WG = W
        
        Xpold = Xp #this is needed because streamflow can only handle regular meshgrids
        xnew = np.linspace(np.min(X), 0, nX)
        Xp, _ = np.meshgrid(xnew, sigma)
        
        U  = griddata((np.ravel(sigmap), np.ravel(Xpold)), np.ravel(U) , (sigmap, Xp), method='linear')
        UR = griddata((np.ravel(sigmap), np.ravel(Xpold)), np.ravel(UR), (sigmap, Xp), method='linear')
        UG = griddata((np.ravel(sigmap), np.ravel(Xpold)), np.ravel(UG), (sigmap, Xp), method='linear')
        UW = griddata((np.ravel(sigmap), np.ravel(Xpold)), np.ravel(UW), (sigmap, Xp), method='linear')
        W  = griddata((np.ravel(sigmap), np.ravel(Xpold)), np.ravel(W) , (sigmap, Xp), method='linear')

        fig, axs = plt.subplots(2,3)
        plt.tight_layout()
        WR = np.zeros_like(UR)
        magR = np.sqrt(UR**2 + WR**2)
        axs[0,0].streamplot(Xp, sigmap, UR, WR, density=1, color='k', linewidth=3*magR/(magR.max()+1e-9))
        for i in range(3):
            axs[0,0].plot(X[Xm[i]]+UR[:,Xm[i]]*np.max(np.abs(X))/10/np.max(np.abs(UR[:,Xm[i]])), sigma)
            axs[0,0].plot(X[Xm[i]]+np.zeros_like(sigma), sigma, 'r:')
        axs[0,0].title.set_text('River-driven')
        axs[0,0].set_xlabel('X')
        axs[0,0].set_ylabel(r'$\sigma$')
        #cbar1 = fig.colorbar(a00, ax=axs[0][1])

        magG = np.sqrt(UG**2 + WG**2)
        axs[0,1].streamplot(Xp, sigmap, UG, WG, density=1, color='k', linewidth=3*magG/(magG.max()+1e-9))
        for i in range(3):
            axs[0,1].plot(X[Xm[i]]+UG[:,Xm[i]]*np.max(np.abs(X))/10/np.max(np.abs(UG[:,Xm[i]])), sigma)
            axs[0,1].plot(X[Xm[i]]+np.zeros_like(sigma), sigma, 'r:')
        axs[0,1].title.set_text('Density-driven')
        axs[0,1].set_xlabel('X')
        axs[0,1].set_ylabel(r'$\sigma$')

        WW = np.zeros_like(UR)
        magW = np.sqrt(UW**2 + WW**2)
        axs[0,2].streamplot(Xp, sigmap, UW, WW, density=1, color='k', linewidth=3*magW/(magW.max()+1e-9))
        for i in range(3):
            axs[0,2].plot(X[Xm[i]]+UW[:,Xm[i]]*np.max(np.abs(X))/10/np.max(np.abs(UW[:,Xm[i]]+1e-9)), sigma)
            axs[0,2].plot(X[Xm[i]]+np.zeros_like(sigma), sigma, 'r:')
        axs[0,2].title.set_text('Wind-driven')
        axs[0,2].set_xlabel('X')
        axs[0,2].set_ylabel(r'$\sigma$')

        Wbar = np.zeros_like(Ubar)
        magW = np.sqrt(Ubar**2 + Wbar**2)
        axs[1,0].streamplot(Xp, sigmap, Ubar, Wbar, density=1, color='k', linewidth=3*magW/(magW.max()+1e-9))
        for i in range(3):
            axs[1,0].plot(X[Xm[i]]+Ubar[:,Xm[i]]*np.max(np.abs(X))/10/np.max(np.abs(Ubar[:,Xm[i]]+1e-9)), sigma)
            axs[1,0].plot(X[Xm[i]]+np.zeros_like(sigma), sigma, 'r:')
        axs[1,0].title.set_text('z-averaged flow')
        axs[1,0].set_xlabel('X')
        axs[1,0].set_ylabel(r'$\sigma$')

        mag = np.sqrt(U**2 + W**2)
        axs[1,1].streamplot(Xp, sigmap, U, W, density=1, color='k', linewidth=3*mag/mag.max())
        for i in range(3):
            axs[1,1].plot(X[Xm[i]]+U[:,Xm[i]]*np.max(np.abs(X))/10/np.max(np.abs(U[:,Xm[i]])), sigma)
            axs[1,1].plot(X[Xm[i]]+np.zeros_like(sigma), sigma, 'r:')
        axs[1,1].title.set_text('Total flow')
        axs[1,1].set_xlabel('X')
        axs[1,1].set_ylabel(r'$\sigma$')
        
        cmap = plt.cm.get_cmap("jet")
        flowmag = axs[1,2].contourf(Xp, sigmap, mag, 50, cmap = cmap)
        axs[1,2].title.set_text('Total flow magnitude')
        axs[1,2].set_xlabel('X')
        axs[1,2].set_ylabel(r'$\sigma$')
        plt.colorbar(flowmag, ax=axs[1,2])    
        return
    
    def plotSimulation(self):
        titleList = [r'$\bar{\Sigma}_X - \bar{\Sigma}$', r'$\bar{\Sigma} - D_3$, where $D_2$ = ' + f'{float(self.D2):.2e}',
                    r'$r - X$', r'$r - \bar{\Sigma}$']
        xVar = [self.Sb_X, self.Sb, self.r, self.r]
        yVar = [self.Sb, self.D3, self.X, self.Sb]
        xLab = [r'$\bar{\Sigma}_X$', r'$\bar{\Sigma}$', r'$r$', r'$r$']
        yLab = [r'$\bar{\Sigma}$', r'$D_3$', r'$X$', r'$\bar{\Sigma}$',]
        #cmap = plt.cm.get_cmap("jet")
        
        Sb_Xplot = np.linspace(-2, 2, 201)
        Sbplot = np.polyval([1,self.b,self.c,0], Sb_Xplot)
        
        a2 = -27
        b2 = 4*self.b**3 - 18*self.b*self.c
        #c2 = self.b**2*self.c**2 - 4*self.c**3
        
        xmax = -b2/(2*a2)
        Sbplot2 = np.linspace(min(-2, xmax), max(xmax, 2), 200)
        D3plot = computeD3(1, self.b, self.c, -Sbplot2)
        
        
        _, axs = plt.subplots(2,2)
        plt.tight_layout()
        cp = []
        for i, ax in enumerate(axs.flatten()):
            cp.append(ax.plot(xVar[i], yVar[i]))
            if i in [0]:
                ax.plot(Sb_Xplot, Sbplot, linestyle = 'dotted')
                ax.plot(self.Exx, self.Exy, marker = 'o', markersize = 6, linestyle = 'dotted')
            if i in [1]:
                ax.plot(Sbplot2, D3plot, linestyle = 'dotted')
            ax.title.set_text(titleList[i])
            ax.set_xlabel(xLab[i])
            ax.set_ylabel(yLab[i])
            ax.grid(c='k', ls='-', alpha=0.3)
        return
        
    def solveODE_Euler(self, ns, opt):
        b, c,  X_a, Sb_X0, Sb_0 = self.b, self.c, self.Xa , self.Sb_X0, self.Sb_0   #The following is only used to determine X interval, not in computation.
        X = np.linspace(X_a, 0, ns)
        Sb, Sb_X = np.zeros_like(X), np.zeros_like(X)
        Sb[-1] = Sb_0
        Sb_X[-1] = Sb_X0
        dX = np.abs(X[2]-X[1])
        p = np.array([1,b,c,-1]) #Coefficients of cubic equation
        if opt in ['minusOne', 'max', 'min']:
            for j in range(ns-2, -1, -1):
                #print(j)
                Sb[j] = Sb[j+1] - dX*Sb_X[j+1]
                p[3] = -Sb[j]
                Sb_X[j] = solveCubic(p, opt)
        elif opt in ['follow']:
            Sb_XR, Sb_XL = self.Exx[0], self.Exx[1]
            Sb_R, Sb_L = self.Exy[0], self.Exy[1]
            branch = 1
            for j in range(ns-2, -1, -1):
                Sb[j] = Sb[j+1] - dX*Sb_X[j+1]
                p[3] = -Sb[j]
                if Sb_X[j+1] > Sb_XR:
                    Sb_X[j] = solveCubic(p, 'max')
                    branch = 1
                elif Sb_X[j+1] >= Sb_XL and Sb_X[j+1] <= Sb_XR:
                    Sb_X[j] = solveCubic(p, 'middle')
                elif Sb_X[j+1] < Sb_XL:
                    Sb_X[j] = solveCubic(p, 'min')
        Sb_XX = Sb_X/(3*Sb_X**2 + 2*b)
        self.Xe, self.Sbe, self.Sb_Xe, self.Sb_XXe = X, Sb, Sb_X, Sb_XX
        #self.X, self.Sb, self.Sb_X, self.Sb_XX = X, Sb, Sb_X, Sb_XX
        return
    '''
        