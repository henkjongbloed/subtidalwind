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
                n = [1, 1001, 401, 1] )#Setting parameters such as BC, alpha. n = [n0D, n1D, n2D, n3D] determines the number of pixels in parameter sensitivity plots.

pp = plotParameters(hatches = False,
        mask = ~gp['mixAway'],
        We = False)


## Figs Results paragraph 1
'''

Ras = [1000, 5e4]
Fws = [1.7, -.5]

for ra, fw in zip(Ras, Fws):
    ndd = makeNDDict(gp, Ra = ra, Fr = 0.025, Fw = fw)
    SM = SingleNonDim(gp, ndd).run()
    plotModel3(pp,SM)
plt.show()

## Figs Results paragraph 2

#Regimes
PSList = []

i = 0

for fr in [0.025]:
#for fr in [0.025]:        
        ndd = makeNDDict(gp, 'Fw', 'Ra', Fr = fr, name = 'fr')
        PSList.append(ParameterSweep(gp, ndd, 0).run())
        plotNDim(pp, PSList[i])
        i += 1
#plotReg4(pp, PSList)

#Regimes with Phi0, SbX0 and Lambda_s
PSList = []
i = 0
for fr in [0.00025, 0.0025, 0.025, 0.1]:
        ndd = makeNDDict(gp, 'Fw', 'Ra', Fr = fr, name = 'fr')
        PSList.append(ParameterSweep(gp, ndd, 0).run())
        #plotNDim(pp, PSList[i])
        i += 1

plotReg4(pp, PSList)


plt.show()
## Appendix Fig
# 

Fwvec =  [-8, -2, -1, -.5, -.25, 0, .25, .5,  1, 2, 8]   #symlog10(np.linspace(invsymlog10(-10), invsymlog10(10), 11))
Ravec = 1e4*np.ones_like(Fwvec)

SM = [SingleNonDim(gp, makeNDDict(gp, Ra = ra, Fr = 2e-2, Fw = fw)).run() for ra,fw in zip(Ravec,Fwvec)]
plotCubics(pp,SM)

## Figs Discussion

PSList = []

i = 0

PSDList, PSList, dimDictList=[],[],[]
gp['mixAway'] = False
q = [2000, 2000, 2000]
h = [10, 18 , 50]
kh = [160, 160, 160]
km = [2e-2, 2e-2, 2e-2]
twl = [[-.5, 8], [-.5, 8], [-.5, 8]]
si = [0,1,2]
sv = [0, 1e-5, 2e-4]

for i in si:
        for j in sv:
                dd, ndd = makeDicts(gp, 'tau_w', Q = q[i], K_H = kh[i], K_M = km[i],H=h[i], tauwLim = twl[i], Sverdrup = j, Guha = False)
                PSDList.append(ParameterSweep(gp, ndd, 1).run())
                dimDictList.append(dd)
        #print(f"Regimes: H = {dimDictList[i]['H']}.")
#plotDim(pp,PS,dd)

gp['mixAway'] = False
for i in si:
        nddN = makeNDDict(gp, 'Fw', 'Ra', Fr = PSDList[3*i].Fr[0], name = 'fr')
        PSList.append(ParameterSweep(gp, nddN, 1).run())
        
        
plotDimNDim(pp, PSDList, dimDictList, PSList)
'''

# Cartoon
PSList = []


SMList = []
nddList = []
Ras = [100, 10000, 100, 5e4]
Fws = [.1,-.1, 2, -.6]
i = 0
for ra, fw in zip(Ras, Fws):
    nddList.append(makeNDDict(gp, Ra = ra, Fr = 0.05, Fw = fw))
    SMList.append(SingleNonDim(gp, nddList[i]).run())
    i+=1

plotSM4(pp,SMList)

plt.show()