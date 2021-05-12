import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from SModelfun import SingleNonDim
from matplotlib import ticker, cm
from PSweepfun import ParameterSweep
from GeneralModelFun import dim2nondim, plotModel, plotSpace, symlog10, invsymlog10, plotDim, plotNDim, globalParameters, plot3D, Ralston, Sverdrup, makeDicts, makeNDDict, plotSModel
import matplotlib.colors as co
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
import time
import warnings

np.seterr(divide = 'ignore') 
warnings.filterwarnings('ignore')
gp = globalParameters() #Setting parameters such as BC, alpha

#ndd = makeNDDict(gp['n'][3]*np.array([1,1,1]))

ndd = makeNDDict(gp, 'Ra', 'Fr', 'Fw' )
PS = ParameterSweep(gp,ndd,0).run()
plotNDim(PS)
plt.show()
'''
ndd = makeNDDict(gp, 'Ra','Fw',Fr = 1e-2)
PS = ParameterSweep(gp,ndd,0).run()
plotNDim(PS)
plt.show()

dd, ndd = makeDicts(gp,'H', 'K_M', tau_w = .01)
PS = ParameterSweep(gp,ndd,1).run()
plotDim(PS, dd)
# Hello World!
'''
ndd = makeNDDict(gp)
SM = SingleNonDim(gp, ndd).run()
plotSModel(SM)
#ndd2 = makeNDDict(gp)
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
plt.show()