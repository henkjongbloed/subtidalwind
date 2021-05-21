import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from SModelfun import SingleNonDim
from matplotlib import ticker, cm
from PSweepfun import ParameterSweep
from GeneralModelFun import Ft2Ra, plotModel, plotSpace, symlog10, invsymlog10, plotDim, plotNDim, globalParameters, plot3D, Ralston, Sverdrup, makeDicts, makeNDDict, plotSModel
import matplotlib.colors as co
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
import time
import warnings
import sys


np.seterr(divide = 'ignore') 
#warnings.filterwarnings('ignore')
plt.rcParams['axes.xmargin'] = 0
gp = globalParameters(R = 2, Ori = True, nps = [1, 101, 21, 21]) #Setting parameters such as BC, alpha

'''
dd, ndd = makeDicts(gp, 'tau_w')
PS = ParameterSweep(gp,ndd,1).run()
plotDim(PS, dd)
# Hello World!

dd, ndd = makeDicts(gp, 'tau_w', 'Q')
PS = ParameterSweep(gp, ndd, 1).run()
plotDim(PS, dd)

# Hello World!
'''
for q in np.logspace(0,3,2):
    dd, ndd = makeDicts(gp, 'tau_w', 'H', Q = q)
    PS = ParameterSweep(gp, ndd, 1).run()
    plotDim(PS, dd)
    
dd, ndd = makeDicts(gp, 'tau_w')
PS = ParameterSweep(gp, ndd, 1).run()
plotDim(PS, dd)

Est = ['Vellar','Columbia', 'James', 'Tees', 'Southampton Waterway', 'Tay', 'Mersey Narrows', 'Bristol Channel' ]
FrGuha = [1.27, .026, .004, .014, .0012, .014, .0009, .006]
FtGuha = [9.0, 5.3, 4.7, 8.9, 5.2, 27.0, 6.7, 27.0] #this is actually their F-tilde

for (fr, ft, name) in zip(FrGuha, FtGuha, Est):
    ndd = makeNDDict(gp, 'Fw', 'Fr', Ra = Ft2Ra(ft), name = name)
    PS = ParameterSweep(gp, ndd, 0).run()
    plotNDim(PS)

for (fr, ft, name) in zip(FrGuha, FtGuha, Est):
    ndd = makeNDDict(gp, 'Fw', Fr = fr, Ra = Ft2Ra(ft), name = name)
    PS = ParameterSweep(gp, ndd, 0).run()
    plotNDim(PS)



plt.show()

#Est = []
#


'''
expar = ['']
EX, dd = [], 10*[None]
for ex in range(10):
    dd[ex], ndd = makeDict(gp, 'H', Q = (ex**2+1)*10, tau_w = .02)
    EX.append(ParameterSweep(gp, ndd, 1))

EXr = [ex.run() for ex in EX]
[plotSmodel(exr, ddr) for exr,ddr in zip(EXr,dd)]

expar = ['']
EX,dd = [], 10*[None]
for ex in range(10):
    dd[ex], ndd = makeDict(gp, 'H', Q = (ex**2+1)*10, tau_w = .02)
    EX.append(ParameterSweep(gp, ndd, 1))

EXr = [ex.run() for ex in EX]
[plotDim(exr, ddr, 'H') for exr,ddr in zip(EXr,dd)]

#PS3.run()
'''
#plt.show()