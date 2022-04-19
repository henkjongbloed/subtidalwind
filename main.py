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
                n = [1, 101, 41, 1] )#Setting parameters such as BC, alpha. n = [n0D, n1D, n2D, n3D] determines the number of pixels in parameter sensitivity plots.

pp = plotParameters(hatches = False,
        mask = ~gp['mixAway'],
        We = False,
        cartoon = 1)


## Discussion Fig (final version)

PSList = []

i = 0

PSDList, PSList, dimDictList=[],[],[]
gp['mixAway'] = False
q = [1000, 1000, 1000]
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



plt.show()