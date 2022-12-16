import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from SModelfun import SingleNonDim
from matplotlib import ticker, cm
from PSweepfun import ParameterSweep
from generalFun import *
from plotFun import *
import matplotlib.colors as co
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager
#import time
#import warnings
#import sys

np.seterr(divide = 'ignore') 
matplotlib.rc('text', usetex=True) #use latex for text

plt.rcParams.update({#'text.usetex' : True,
        'font.size' : 11,
        'font.family' : 'DeJavu Serif',
        'font.serif' : 'Computer Modern',
        'axes.xmargin' : 0,
        'hatch.color': 'gray',
        }) 

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

gp = globalParameters(R = 2, 
                ep = 1/30, 
                Ori = True, 
                m = 1.01, 
                mixAway = False,
                realTransport = True,
                scaledTransportPlot = True,
                tolUN = 0,
                tolNEG = 0,
                n = [1, 11, 11, 1] )#Setting parameters such as BC, alpha. n = [n0D, n1D, n2D, n3D] determines the number of pixels in parameter sensitivity plots.

pp = plotParameters(hatches = False,
        mask = ~gp['mixAway'],
        We = False,
        cartoon = 1)

# Figures for Ems Scheldt workshop

Est = ['Delaware', 'Scheldt', 'Ems']

Ua = np.array([5, -5, -1.5]) #Annual mean wind velocity
twl = 2.6e-3*1.225*Ua**2*np.sign(Ua)
PSList = []

i = 0

PSDList, PSList, dimDictList, PSSList, datList=[],[],[],[],[]
gp['mixAway'] = False
#q = [1000, 1000, 1000]
h = np.array([20, 10, 7])

s_0 = 30.0
g = 9.81
beta = 7.6e-4
rho_0 = 1000.0
B = np.array([1000.0, 1000.0, 10000.0])
c = np.sqrt(s_0*g*beta*h)
sv = np.array([0, 4.1e-5, 8.1e-5])/(1.225*2.6e-3) # Sverdrup parameter omega
kmw = np.array([3e-3, 2.2e-2, 1e-3])

FrD = np.array([.02, 1.7e-2, .005/h[2]/c[2]]) #FrDijkstra
RaD = np.array([1e4, 30, c[2]**2*h[2]**2/(kmw[2]*100)]) #RaDijkstra: RaD  = Raw



q = FrD*(B*h*c) # To obtain correct value for Fr

print(q)
print(FrD)
print(RaD)
twlim = [np.array([-np.abs(tw), np.abs(tw)]) for tw in twl]
si = [0,1,2]

gp['mixAway'] = False

for i in si: # system
        for j in range(len(sv)): #value of omega
                km0 = kmw[i] - sv[j]*np.abs(twl[i])
                # print(km0)
                Ra0 = RaD[i]*kmw[i]/km0
                kh = c[i]**2*h[i]**2/(Ra0*km0) # To obtain correct value of Ra
                if i==1:
                        kh = kh
                dd, ndd = makeDicts(gp, 'tau_w', Q = q[i], K_H = kh, K_M = km0, H = h[i], tauwLim = 2.5**2*twlim[i], Sverdrup = sv[j], Guha = False) #tau_w-lim: Amplification factor
                dddat, ndddat = makeDicts(gp, tau_w = twl[i], Q = q[i], K_H = kh, K_M = kmw[i], H = h[i]) #For the scatter points indicating data values
                
                PSDList.append(ParameterSweep(gp, ndd, 1).run())
                PSSList.append(SingleNonDim(gp, ndddat).run()) # Only needed for circle plots indicating estuaries in parameter space. Needed: Fr, Ra, Fw, u_a
                dimDictList.append(dd)
                datList.append([Ua[si[i]], dddat])
                plotModel3(pp, PSSList[-1])
        #print(f"Regimes: H = {dimDictList[i]['H']}.")
#plotDim(pp,PS,dd)

gp['mixAway'] = False
for i in si:
        nddN = makeNDDict(gp, 'Fw', 'Ra', Fr = PSDList[3*i].Fr[0], name = Est[i])
        PSList.append(ParameterSweep(gp, nddN, 1).run())
        
        
plotDimNDimData(pp, PSDList, dimDictList, PSList, PSSList, datList)

plt.show()