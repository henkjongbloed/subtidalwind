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
from matplotlib.widgets import Slider, Button, RadioButtons

np.seterr(divide = 'ignore') 
warnings.filterwarnings('ignore')
gp = globalParameters(SMGrid = [51, 11]) #Setting parameters such as BC, alpha

# The parametrized function to be plotted
def f(gp, Ra, Fr, Fw):
    ndd = makeNDDict(gp, Ra = Ra, Fr = Fr, Fw = Fw)
    SM0 = SingleNonDim(gp, ndd).run()
    return SM0

# Define initial parameters
ndd = makeNDDict(gp)
SM0 = SingleNonDim(gp, ndd).run()

Ra0 = ndd['Ra']
Fr0 = ndd['Fr']
Fw0 = ndd['Fw']
#global cf
#global cg
# Create the figure and the line that we will manipulate

fig, axs = plt.subplots(2, 2)
SM0 = f(gp, Ra0, Fr0, Fw0)
X, Xp, sigmap, sigma = SM0.X, SM0.Xp, SM0.sigmap, SM0.sigma
S, Sb = SM0.S, SM0.Sb

Sb_X, r = SM0.Sb_X, SM0.r

col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

U, W = SM0.U, SM0.W
T = SM0.T

labT = ['G-G', 'G-R', 'G-W', 'R-R', 'R-W', 'W-W', 'DI', '|FL|']
cmap = 'Blues'

#plt.tight_layout()

axs[0,0].plot(X, Sb, 'k', label = 'Averaged')
axs[0,0].plot(X, S[-1,:], 'k:', label = 'Surface') #S[-1,:] is landward surface, S[0,:] is seaward bottom (REVERSED)
axs[0,0].plot(X, S[0,:], 'k-.', label = 'Bottom')
axs[0,0].plot(X, S[0,:] - S[-1,:], 'k--', label = 'Stratification')    
axs[0,0].title.set_text('Averaged salinity')
axs[0,0].set_xlabel(r'$X$')
axs[0,0].set_ylabel(r'$\bar{\Sigma}$')
axs[0,0].legend()
axs[0,0].grid(True)

f1 = axs[1,0].contourf(Xp, sigmap, S, 50, cmap=cmap, corner_mask = True)
#axs[1,0].contour(f1, levels = np.linspace(0,1,10), colors = 'k', linewidths = 0.5)
axs[1,0].contour(f1, levels = np.linspace(0,1,15), colors = 'k', linewidths = 0.5, alpha = 0.2)
axs[1,0].title.set_text(f'Salinity: Negative = {SM0.maskNEG} and Unstable = {SM0.maskIS}')
axs[1,0].set_xlabel(r'$X$')
axs[1,0].set_ylabel(r'$\sigma$')
if np.amin(S) < 0:
    axs[1,0].contourf(Xp, sigmap, S, levels = [np.amin(S), 0, np.amax(S)], cmap = cmap, corner_mask = True, hatches=["//", ""], alpha = 0)
    axs[1,0].contour(f1, levels = [0], colors='w', linewidths = 1.5)
#plt.colorbar(f1, ax=axs[1,0])
    
for t in range(len(T)):
    if t<7:
        axs[0,1].plot(X, T[t], label = labT[t], color = col[t])
    else:
        axs[0,1].plot(X, -T[t], '-.', label = labT[t], color = col[t])
axs[0,1].title.set_text('Salt transport')
axs[0,1].set_xlabel(r'$X$')
axs[0,1].set_ylabel('Relative contribution')
axs[0,1].legend()
axs[0,1].grid(True)

sbxmin = min([SM0.Exx[1], np.min(Sb_X)])
sbxmax = max([SM0.Exx[0], np.max(Sb_X)])

Sb_Xplot = np.linspace(sbxmin, sbxmax, 201)
Sbplot = np.polyval([1,SM0.b,SM0.c,0], Sb_Xplot)

axs[1,1].plot(Sb_Xplot, Sbplot, ls = 'dotted', label = 'Curve')
axs[1,1].plot(Sb_X, Sb, lw = 2, label = 'Realised')
axs[1,1].scatter(SM0.Exx, SM0.Exy, marker = 'o', label = 'J = 0 or H = 0')
axs[1,1].title.set_text(r'$\bar{\Sigma}_X - \bar{\Sigma}$ Curve - ' + f'Non-unique = {SM0.maskNU}')
axs[1,1].set_xlabel(r'$\bar{\Sigma}_X$')
axs[1,1].set_ylabel(r'$\bar{\Sigma}$')
axs[1,1].grid(True)
axs[1,1].legend()

## Include Sliders
del SM0

axcolor = 'lightgoldenrodyellow'
axs[0,0].margins(x=0)
axs[1,0].margins(x=0)
axs[0,1].margins(x=0)
axs[1,1].margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left = 0.25)

# Make a horizontal slider to control the frequency.
axRa = plt.axes([0.05, 0.15, 0.025, 0.7], facecolor=axcolor)
Ra_s = Slider(
    ax = axRa,
    label = 'Ra',
    valmin = 0,
    valmax = 5,
    valinit = np.log10(Ra0),
    valstep = .2,
    orientation = "vertical",
)
axFr = plt.axes([0.1, 0.15, 0.025, 0.7], facecolor=axcolor)
Fr_s = Slider(
    ax = axFr,
    label = 'Fr',
    valmin = -4,
    valmax = 0,
    valinit = np.log10(Fr0),
    valstep = .2,
    orientation = "vertical",

)
axFw = plt.axes([0.15, 0.15, 0.025, 0.7], facecolor=axcolor)
Fw_s = Slider(
    ax = axFw,
    label = 'Fw',
    valmin = -2,
    valmax = 2,
    valinit = symlog10(Fw0),
    valstep = .2,
    orientation = "vertical",
)

# The function to be called anytime a slider's value changes
def update(val):
    #print(val)
    #Ra, Fr, Fw = ,, 
    #print([Ra, Fr, Fw])
    ndd = makeNDDict(gp, Ra = 10**Ra_s.val, Fr =  10**Fr_s.val, Fw = invsymlog10(Fw_s.val))
    SM = SingleNonDim(gp, ndd).run()
    
    #SM = f(gp, Ra, Fr, Fw)
    X, Xp, sigmap = SM.X, SM.Xp, SM.sigmap
    S, Sb = SM.S, SM.Sb

    Sb_X = SM.Sb_X
    #print(Sb_X)
    col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    #U, W = SM.U, SM.W
    T = SM.T

    labT = ['G-G', 'G-R', 'G-W', 'R-R', 'R-W', 'W-W', 'DI', '|FL|']
    cmap = 'Blues'

    #plt.tight_layout()

    axs[0,0].clear()
    axs[0,0].plot(X, Sb, 'k', label = 'Averaged')
    axs[0,0].plot(X, S[-1,:], 'k:', label = 'Surface') #S[-1,:] is landward surface, S[0,:] is seaward bottom (REVERSED)
    axs[0,0].plot(X, S[0,:], 'k-.', label = 'Bottom')
    axs[0,0].plot(X, S[0,:] - S[-1,:], 'k--', label = 'Stratification')    
    axs[0,0].title.set_text('Averaged salinity')
    axs[0,0].set_xlabel(r'$X$')
    axs[0,0].set_ylabel(r'$\bar{\Sigma}$')
    axs[0,0].legend()
    axs[0,0].grid(True)
    
    axs[1,0].clear()
    f1 = axs[1,0].contourf(Xp, sigmap, S, 50, cmap=cmap, corner_mask = True)
    #axs[1,0].contour(f1, levels = np.linspace(0,1,10), colors = 'k', linewidths = 0.5)
    axs[1,0].contour(f1, levels = np.linspace(0,1,15), colors = 'k', linewidths = 0.5, alpha = 0.2)
    axs[1,0].title.set_text(f'Salinity: Negative = {SM.maskNEG} and Unstable = {SM.maskIS}')
    axs[1,0].set_xlabel(r'$X$')
    axs[1,0].set_ylabel(r'$\sigma$')
    if np.amin(S) < 0:
        axs[1,0].contourf(Xp, sigmap, S, levels = [np.amin(S), 0, np.amax(S)], cmap = cmap, corner_mask = True, hatches=["//", ""], alpha = 0)
        axs[1,0].contour(f1, levels = [0], colors='w', linewidths = 1.5)
    #plt.colorbar(f1, ax=axs[1,0])
    
    axs[0,1].clear()
    for t in range(len(T)):
        if t<7:
            axs[0,1].plot(X, T[t], label = labT[t], color = col[t])
        else:
            axs[0,1].plot(X, -T[t], '-.', label = labT[t], color = col[t])
    axs[0,1].title.set_text('Salt transport')
    axs[0,1].set_xlabel(r'$X$')
    axs[0,1].set_ylabel('Relative contribution')
    axs[0,1].legend()
    axs[0,1].grid(True)

    sbxmin = min([SM.Exx[1], np.min(Sb_X)])
    sbxmax = max([SM.Exx[0], np.max(Sb_X)])

    Sb_Xplot = np.linspace(sbxmin, sbxmax, 101)
    Sbplot = np.polyval([1,SM.b,SM.c,0], Sb_Xplot)
    
    axs[1,1].clear()
    axs[1,1].plot(Sb_Xplot, Sbplot, ls = 'dotted', label = 'Curve')
    axs[1,1].plot(Sb_X, Sb, lw = 2, label = 'Realised')
    axs[1,1].scatter(SM.Exx, SM.Exy, marker = 'o', label = 'J = 0 or H = 0')
    axs[1,1].title.set_text(r'$\bar{\Sigma}_X - \bar{\Sigma}$ Curve - ' + f'Non-unique = {SM.maskNU}')
    axs[1,1].set_xlabel(r'$\bar{\Sigma}_X$')
    axs[1,1].set_ylabel(r'$\bar{\Sigma}$')
    axs[1,1].grid(True)
    axs[1,1].legend()
    #
    #fig.canvas.draw_idle()
    del SM


# register the update function with each slider
Ra_s.on_changed(update)
Fr_s.on_changed(update)
Fw_s.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    Ra_s.reset()
    Fr_s.reset()
    Fw_s.reset()
button.on_clicked(reset)

plt.show()

'''
def compute_plot():
    X, Xp, sigmap = SM.X, SM.Xp, SM.sigmap, SM.sigma
    S, Sb = SM.S, SM.Sb
    
    Sb_X, r = SM.Sb_X, SM.r
    
    col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    
    U, W = SM.U, SM.W
    T = SM.T

    labT = ['G-G', 'G-R', 'G-W', 'R-R', 'R-W', 'W-W', 'DI', '|FL|']
    cmap = 'Blues'

    _, axs = plt.subplots(2,2)
    plt.tight_layout()
    
    
    axs[0,0].plot(X, Sb, 'k', label = 'Averaged')
    axs[0,0].plot(X, S[-1,:], 'k:', label = 'Surface') #S[-1,:] is landward surface, S[0,:] is seaward bottom (REVERSED)
    axs[0,0].plot(X, S[0,:], 'k-.', label = 'Bottom')
    axs[0,0].plot(X, S[0,:] - S[-1,:], 'k--', label = 'Stratification')    
    axs[0,0].title.set_text('Averaged salinity')
    axs[0,0].set_xlabel(r'$X$')
    axs[0,0].set_ylabel(r'$\bar{\Sigma}$')
    axs[0,0].legend()
    axs[0,0].grid(True)
    
    f1 = axs[1,0].contourf(Xp, sigmap, S, 50, cmap=cmap, corner_mask = True)
    #axs[1,0].contour(f1, levels = np.linspace(0,1,10), colors = 'k', linewidths = 0.5)
    #axs[1,0].contour(f1, levels = np.linspace(0,1,50), colors = 'k', linewidths = 0.5, alpha = 0.2)
    axs[1,0].title.set_text(f'Salinity: Negative = {SM.maskNEG} and Unstable = {SM.maskIS}')
    axs[1,0].set_xlabel(r'$X$')
    axs[1,0].set_ylabel(r'$\sigma$')
    if np.amin(S) < 0:
        axs[1,0].contourf(Xp, sigmap, S, levels = [np.amin(S), 0, np.amax(S)], cmap = cmap, corner_mask = True, hatches=["//", ""], alpha = 0)
        axs[1,0].contour(f1, levels = [0], colors='w', linewidths = 1.5)
    plt.colorbar(f1, ax=axs[1,0])
        
    for t in range(len(T)):
        if t<7:
            axs[0,1].plot(X, T[t], label = labT[t], color = col[t])
        else:
            axs[0,1].plot(X, -T[t], '-.', label = labT[t], color = col[t])
    axs[0,1].title.set_text('Salt transport')
    axs[0,1].set_xlabel(r'$X$')
    axs[0,1].set_ylabel('Relative contribution')
    axs[0,1].legend()
    axs[0,1].grid(True)
    
    sbxmin = min([SM.Exx[1], np.min(Sb_X)])
    sbxmax = max([SM.Exx[0], np.max(Sb_X)])

    Sb_Xplot = np.linspace(sbxmin, sbxmax, 201)
    Sbplot = np.polyval([1,SM.b,SM.c,0], Sb_Xplot)
    
    axs[2,1].plot(Sb_Xplot, Sbplot, ls = 'dotted', label = 'Curve')
    axs[2,1].plot(Sb_X, Sb, lw = 2, label = 'Realised')
    axs[2,1].scatter(SM.Exx, SM.Exy, marker = 'o', label = 'J = 0 or H = 0')
    axs[2,1].title.set_text(r'$\bar{\Sigma}_X - \bar{\Sigma}$ Curve - ' + f'Non-unique = {SM.maskNU}')
    axs[2,1].set_xlabel(r'$\bar{\Sigma}_X$')
    axs[2,1].set_ylabel(r'$\bar{\Sigma}$')
    axs[2,1].grid(True)
    axs[2,1].legend()
'''