import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from SModelfun import SingleNonDim
from PSweepfun import ParameterSweep
#from GeneralModelFun import formFunctions, globalParameters, symlog10, invsymlog10
import matplotlib.colors as co

import matplotlib as mpl
import matplotlib.font_manager

params = {#'text.usetex' : True,
        'font.size' : 11,
        'font.family' : 'DeJavu Serif',
        'font.serif' : 'Utopia',
        'axes.xmargin' : 0,
        }
plt.rcParams.update(params)

x = np.linspace(0,10,100)
y = x**2
fig = plt.figure(constrained_layout=False)
s = fig.add_gridspec(ncols=2, nrows=2)

for a in s:
    print(a)
    ax = fig.add_subplot(a)
    ax.plot(x,y)
    ax.set_xlabel(r'Hello $x$')
    ax.set_ylabel(r'$x^2$')
    ax.set_title(r'Simple plot $y=x^2$')

plt.show()

#plt.plot(x,y)





'''
sigmap = np.linspace(-1, 0, 201)

*P, PW = formFunctions(sigmap, gp['R'])

P[1] = 48*P[1]
P[4] = 48*P[4]

fig = plt.figure(constrained_layout=True)

spec5 = fig.add_gridspec(ncols=3, nrows=2)
i = 0
leg = [r'$P_1(\sigma)$', r'$P_2(\sigma)$', r'$P_3(\sigma)$', r'$P_4(\sigma)$', r'$P_5(\sigma)$', r'$P_6(\sigma)$']
ylab = [r'$\sigma$', '', '', r'$\sigma$', '', '']
for row in range(2):
    for col in range(3):
        ax = fig.add_subplot(spec5[row, col])
        ax.plot(P[i], sigmap, c = 'k')
        print(np.mean(P[i]))
        ax.set_title(leg[i])
        ax.set_ylabel(ylab[i])
        ax.grid(True)       
        i+=1

        #label = 'Width: {}\nHeight: {}'.format(widths[col], heights[row])
        #ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')

#i = 1
#for p in P:
    #plt.plot(p, sigmap, label = str(i))
    #i+=1
#plt.legend()
plt.show()
'''