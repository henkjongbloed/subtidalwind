import numpy as np
import numpy.ma as ma
import matplotlib as mp
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors as co
#import matplotlib.colors as colors
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from GeneralModelFun import solveCubic, computeD2, computeD3, dim2nondim, addC, computeLocalExtrema, I0,I1,I2,I3, scoreIJ, computeRegime, computeMask
from matplotlib import ticker

class ParameterSweep:
    def __init__(self, gp, nondimDict, dim):
        self.gp = gp
        self.nondimDict = nondimDict # Governing parameters
        self.dim = dim               # Dimensional or non-dimensional
        self.nps = nondimDict['nps']
        return
    
    def run(self):
        self.inputPS()  # Construct the meshgrid
        self.C = self.gp['C']          # Add coefficients
        self.addODEC()                  # Contstruct b and c, b0, b0, b00, c00
        self.mouthPS()                  # Compute Sb_X0 and Phi_0
        self.nonLinPS()                 # Compute all relevant quantities (masks, )
        return self
    
    def inputPS(self):
        nondimDict = self.nondimDict
        Ra = nondimDict['Ra']
        Fr = nondimDict['Fr']
        Fw = nondimDict['Fw']
        self.Sc = nondimDict['Sc']
        #Ra1, Fr1, Fw1 = Ra.flatten(), Fr.flatten(), Fw.flatten()
        if len(nondimDict['pars']) > 1:
            n = np.prod(nondimDict['nps'])
        else:
            n = nondimDict['nps']
        self.Ra, self.Fr, self.Fw = Ra*np.ones(n), Fr*np.ones(n), Fw*np.ones(n)
        #self.Ra, self.Fr, self.Fw = Ra, Fr, Fw
        return

    def addODEC(self):
        dummy = np.zeros_like(self.Ra)
        C = self.gp['C']
        Ra, Fr, Fw, Sc = self.Ra, self.Fr, self.Fw, self.Sc
        self.b = Sc**(1/3)*(C[7]*Fr**(2/3) + C[8]*Fr**(-1/3)*Fw) + dummy
        self.c = Sc**(2/3)*(C[9]*Fr**(4/3) + C[10]*Fr**(1/3)*Fw + C[11]*Fr**(-2/3)*Fw**2) + C[12]*Sc**(-1/3)*Fr**(-2/3)*Ra**(-1) + dummy
        self.b0 = self.b + C[13]*C[18]*Sc**(1/3)*Fr**(2/3)
        self.c0 = self.c + C[12]*Sc**(2/3)*Fr**(1/3)*(C[17]*Fr + C[19]*Fw)
        self.b00 = C[13]*Sc**(1/3)*Fr**(2/3)*C[15] + dummy
        self.c00 = C[12]*Sc**(2/3)*Fr**(1/3)*(C[14]*Fr +C[16]*Fw) + dummy
        
        self.L = [np.ones_like(dummy), #GG
            C[7]*Sc**(1/3)*Fr**(2/3),#GR squared!
            C[8]*Sc**(1/3)*Fr**(-1/3)*Fw,#GW squared!
            C[9]*Sc**(2/3)*Fr**(4/3),#RR
            C[10]*Sc**(2/3)*Fr**(1/3)*Fw,#RW
            C[11]*Sc**(2/3)*Fr**(-2/3)*Fw**2,#WW
            C[12]*Sc**(-1/3)*Fr**(-2/3)*Ra**-1,#D
            - np.ones_like(dummy)]#F
        return
    
    def mouthPS(self):
        b, c, b0, c0, b00, c00 = self.b, self.c, self.b0, self.c0, self.b00, self.c00
        Sb_X0 = np.zeros_like(self.b)
        m = np.empty_like(Sb_X0, dtype=bool)
        for i in range(len(b)):
            Sb_X0[i], m[i] = solveCubic(np.array([1.0,b0[i],c0[i],-1.0]))
        Sb_X0 = ma.masked_array(Sb_X0, mask = m)
        Sb_0 = Sb_X0**3 + b*Sb_X0**2 + c*Sb_X0
        S_00 = Sb_0 + b00*Sb_X0**2 + c00*Sb_X0
        Phi_0 = 1 - S_00
        self.Sb_X0, self.Phi_0, self.Sb_0, self.maskNU0 = Sb_X0, Phi_0, Sb_0, m
        return
    
    def nonLinPS(self):
        Sb_X0, b, c, alpha, L, maskNU0 =  self.Sb_X0, self.b, self.c, self.gp['alpha'], self.L, self.maskNU0
        dummy = 0.0*Sb_X0
        Xa, ra = ma.array(np.empty_like(dummy), mask = maskNU0), ma.array(np.empty_like(dummy), mask = maskNU0)
        #FIRST MASK: Non-unique Sb_X0 or Non-unique / positive Xa
        for i in range(len(dummy)):  
            if maskNU0[i] == True: #Solution is non unique -> do nothing
                Xa[i] = ma.asanyarray(Xa[i])
                ra[i] = ma.asanyarray(ra[i])
            else:
                rt, _ = solveCubic(np.array([Sb_X0[i]**2, b[i]*Sb_X0[i], c[i], -alpha*(Sb_X0[i]**2 + b[i]*Sb_X0[i] + c[i])])) #rt = exp(-r)
                if rt > 0:
                    ra[i] = -np.log(rt)
                    Xa[i] = 3/2*Sb_X0[i]**2*(np.exp(-2*ra[i])-1) + 2*b[i]*Sb_X0[i]*(np.exp(-ra[i])-1) - c[i]*ra[i]
                else:
                    Xa[i] = ma.asanyarray(Xa[i])
                    ra[i] = ma.asanyarray(ra[i])
                    
                if Xa[i] >= 0: #not physical -> move to other mask
                    Xa[i] = ma.asanyarray(Xa[i])
                    ra[i] = ma.asanyarray(ra[i])
                    maskNU0[i] = True
        Ls = np.abs(Xa)
        #Second mask:
        #C = self.gp['C']
        Ra, Fr, Fw, Sc = self.Ra, self.Fr, self.Fw, self.Sc
        #ns = self.gp
        #maskPS(b,c, Ra, Fr, Fw, Sc, Sb_X0, ra, ns, C)
        
        bl, bu = Sb_X0*np.exp(-ra), Sb_X0
        temp1,temp2,temp3 = np.zeros_like(b), np.zeros_like(b), np.zeros_like(b)
        maskNU, maskUS, maskNEG, maskPRIT = np.empty_like(b, dtype=bool), np.empty_like(b, dtype=bool), np.empty_like(b, dtype=bool), np.empty_like(b, dtype=bool) #make this a boolean array.
        #maskNU: Simulation encounters a local minimum/maximum
        #maskUS: Unstable stratification
        #maskNEG: negative salt
        for i in range(len(b)):
            maskNU[i], maskUS[i], maskNEG[i], maskPRIT[i] = computeMask(b[i],c[i], Ra[i], Fr[i], Fw[i], Sc, Sb_X0[i], ra[i], self.gp)
            temp1[i] = 3*(I3(bu[i],1.0,b[i],c[i])-I3(bl[i],1.0,b[i],c[i])) + 2*b[i]*(I2(bu[i],1.0,b[i],c[i])-I2(bl[i],1.0,b[i],c[i])) + c[i]*(I1(bu[i],1.0,b[i],c[i])-I1(bl[i],1.0,b[i],c[i]))
            temp2[i] = 3*(I2(bu[i],1,b[i],c[i])-I2(bl[i],1,b[i],c[i])) + 2*b[i]*(I1(bu[i],1,b[i],c[i])-I1(bl[i],1,b[i],c[i])) + c[i]*(I0(bu[i],1,b[i],c[i])-I0(bl[i],1,b[i],c[i]))
            temp3[i] = 3*(I2(bu[i],1,b[i],c[i])-I2(bl[i],1,b[i],c[i])) + 2*b[i]*(I1(bu[i],1,b[i],c[i])-I1(bl[i],1,b[i],c[i])) + c[i]*(I0(bu[i],1,b[i],c[i])-I0(bl[i],1,b[i],c[i]))
            #maskUS[i] = False
        totMask = np.logical_or(maskNU, maskUS)
        totMask = np.logical_or(totMask, maskNEG)
        totMask = np.logical_or(totMask, maskPRIT)
        totMask = np.logical_or(totMask, self.maskNU0)
        #totMask2 = maskUS|maskNU|maskNEG
        #plt.plot(maskIS)
        #plt.plot(np.log10(Fr))
        #plt.plot(np.log10(Ra))
        
        aT = [temp1, #Analytically integrated scaled transports (not yet divided by Ls)
            temp2,
            temp3,
            L[0]*np.log(1/alpha), #L[0] is just a dummy variable, no length
            L[0]*np.log(1/alpha),
            L[0]*np.log(1/alpha),
            L[0]*np.log(1/alpha),
            Ls
            ]
        
        aTT = [l*at/Ls for l,at in zip(L,aT)] #aTT contains 8 np.arrays of size nps
        score = np.zeros([4,len(b)])
        Reg, maxTT = np.zeros_like(b), np.zeros_like(b)
        for i in range(len(b)):
            att = np.array([aTT[0][i],aTT[1][i],aTT[2][i],aTT[3][i],aTT[4][i],aTT[5][i],aTT[6][i],aTT[7][i]])
            score[:,i] = scoreIJ(att)
            maxTT[i] = score[3,i] #The index of the maximum transport term (absolute value) (excluding River Flushing)
            Reg[i] = computeRegime(score[0:3,i])
        self.domTerm = [np.count_nonzero(maxTT == ind) for ind in range(7)]
        self.domTermPlot = maxTT
        Sb0 = self.Sb_0
        #theoretical salt intrusion lengths
        self.LSReg = [3/2*Sb0**(2/3)*L[0]*(1-alpha**2/3),
                2*np.sqrt(L[1])*(np.sqrt(Sb0) - alpha**2*Sb0**2), 
                2*np.sqrt(L[2])*(np.sqrt(Sb0) - alpha**2*Sb0**2),
                -L[3]*np.log(Sb0*alpha),
                -L[4]*np.log(Sb0*alpha),
                -L[5]*np.log(Sb0*alpha),
                -L[6]*np.log(Sb0*alpha)] 
        
        #print(self.domTerm)
        #Apply further masks.
        Phi_0 = ma.masked_array(self.Phi_0, mask = totMask)
        Xa = ma.masked_array(Xa, mask = totMask)
        Reg = ma.masked_array(Reg, mask = totMask)
        Ls = ma.masked_array(Ls, mask = totMask)
        self.totMask = totMask
        # Look into this RESHAPE stuff!!
        self.aTT = [it.reshape(self.nps) for it in aTT]
        self.Xa, self.ra, self.Ls = Xa.reshape(self.nps), ra.reshape(self.nps), Ls.reshape(self.nps)
        self.Sb_X0, self.Phi_0, self.Sb_0 = Sb_X0.reshape(self.nps), Phi_0.reshape(self.nps), self.Sb_0.reshape(self.nps)
        self.Reg = Reg.reshape(self.nps)
        if self.dim == 0:
            self.Ra, self.Fr, self.Fw = self.Ra.reshape(self.nps), self.Fr.reshape(self.nps), self.Fw.reshape(self.nps)
        return