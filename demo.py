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

gp = globalParameters(R = 2, # Model parameters
                ep = 1/30, 
                Ori = True, 
                m = 1.01, 
                mixAway = False,
                realTransport = True,
                scaledTransportPlot = True,
                tolUN = 0,
                tolNEG = 0,
                n = [1, 11, 11, 1] )#Setting parameters such as BC, alpha. n = [n0D, n1D, n2D, n3D] determines the number of pixels in parameter sensitivity plots.

pp = plotParameters(hatches = False, # Model parameters
        mask = ~gp['mixAway'],
        We = False,
        cartoon = 1)

dd, ndd = makeDicts(gp, Q = 1000, 
        H = 20,  
        K_M = 1e-3, 
        tau_w = 0,
        K_H = 250)

model = SingleNonDim(gp, ndd)
model.run()

model.make_dimensional(dd) # Requested variables are now added to the model (dimensional quantities, grid and zeta_x)
# After having run the model, properties can be extracted e.g. as follows:
# X grid = model.Xp (nondim)
# sigma grid  = model.sigmap (nondim)

plotModel3(pp, model)

plt.show()