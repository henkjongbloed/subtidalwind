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

plt.rcParams.update({#'text.usetex' : True,
        'font.size' : 11,
        'font.family' : 'DeJavu Serif',
        'font.serif' : 'Computer Modern',
        'axes.xmargin' : 0,
        'hatch.color': 'gray'
        }) 

gp = globalParameters(R = 2, 
                ep = 1/30, 
                Ori = True, 
                m = 1.01, 
                mixAway = False,
                realTransport = True,
                scaledTransportPlot = True,
                tolUN = 0,
                tolNEG = 0,
                n = [1, 601, 501, 21]) #Setting parameters such as BC, alpha

pp = plotParameters(hatches = False,
                    mask = ~gp['mixAway'],
                    We = False)

plt.show()