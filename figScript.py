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
#import time
#import warnings
#import sys


np.seterr(divide = 'ignore') 
#warnings.filterwarnings('ignore')
plt.rcParams['axes.xmargin'] = 0
gp = globalParameters(R = 2, 
                    q = 1/30, 
                    Ori = True, 
                    m = 1.01, 
                    mixAway = False,
                    realTransport = True,
                    scaledTransportPlot = True,
                    tolUN = 0.0,
                    tolNEG = 0.0,
                    n = [1, 51, 15, 21]) #Setting parameters such as BC, alpha


# Generate cubic Curves (gp determines algorithm)
Fwvec =  [-8, -4,-2, -1, -.4, -.1, 0, .1, .4, 1, 2, 4, 8]   #symlog10(np.linspace(invsymlog10(-10), invsymlog10(10), 11))
SM = [SingleNonDim(gp, makeNDDict(gp, Ra = 1e5, Fr = 1e-2, Fw = fw)).run() for fw in Fwvec]
plotCubics(SM)

# Generate Estuaries Dijkstra 2021 Plot
#Est = ['Vellar','Columbia', 'James', 'Tees', 'Southampton Waterway', 'Tay', 'Mersey Narrows', 'Bristol Channel' ]
Est = [ 'Ch', 'Mh', 'Sc', 'R', 'Sy', 'Hh',  'Du', 'Cl', 'Hl', 'Ml', 'Dn', 'J']
FrD = [.3, .7,  9e-3, .2, .25, .15, .2, 4e-2, 1.5e-2, 7e-2, 2e-2, 5e-3]
RaD = [60, 80, 80, 200, 800, 850, 1000, 1800, 3000, 1e4, 1.3e4, 6e4]

selec = [0, 3, 6, 10]
PS = []
i = selec[0]
for (fr, ra, name) in zip(FrD, RaD, Est):
    if i in selec:
        ndd = makeNDDict(gp, 'Fw', Ra = ra, name = name)
        PS.append(ParameterSweep(gp, ndd, 0).run())
    i = i + 1
    #plotNDim(PS)
#plt.show()
plotNDimEst(PS)


# Entire parameter space plot

for fr in [1e-3, 1e-2, 1e-1]:
    ndd = makeNDDict(gp, 'Fw', 'Ra', Fr = fr, name = 'fr')
    PS = ParameterSweep(gp,ndd,0).run()
    plotNDim(PS)
    #print(np.sum(PS.mask[:,4]))
#plt.plot(np.sum(PS.T[:,0:6], axis = 1))

#Res2
PSList = []

i = 0

for fr in [0.00025, 0.0025, 0.025, 0.1]:
        ndd = makeNDDict(gp, 'Fw', 'Ra', Fr = fr, name = 'fr')
        PSList.append(ParameterSweep(gp, ndd, 0).run())
        #plotNDim(pp, PSList[i])
        i += 1

plotReg4(pp, PSList)


PSList = []

i = 0

for fw in [-.25, 0, .25, 1]:
        ndd = makeNDDict(gp, 'Ra', 'Fr', Fw = fw, name = 'fr')
        PSList.append(ParameterSweep(gp, ndd, 0).run())
        #plotNDim(pp, PSList[i])
        i += 1

plotReg4(pp, PSList)
# Sensitivity to epsilon

for q in [1/3, 1/30, 1/300, 1/3000]:
    gp['q'] = q
    #gp['C'] = addC(a)
    for fr in [1e-3, 1e-2, 1e-1, 1e0]:
        ndd = makeNDDict(gp, 'Fw', 'Ra', Fr = fr, name = 'Fr = ' + str(fr))
        PS = ParameterSweep(gp,ndd,0).run()
        plotNDim(PS)
        #print(f'Using a = {a} yields a mean Phi0 of {np.mean(PS.Phi_0):.2f}')
        #print(f'Using a = {a} yields a mean LS of {np.mean(np.abs(PS.Xs)):.0f}')

        #print(np.sum(PS.mask[:,4]))
        
# Guha script

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


# Make the four SingleNonDim plots

Ras = [100, 1e5, 1e4, 5e5, 1e3]
Fws = [0.17, .03, 1.7, -2, .3]

for ra, fw in zip(Ras, Fws):
    ndd = makeNDDict(gp, Ra = ra, Fr = 1e-2, Fw = fw)
    SM = SingleNonDim(gp, ndd).run()
    plotModel3(SM)


# Inspect largest unstable strat
ndd = makeNDDict(gp, Ra = 1e5, Fr = 1e-2, Fw = -1)
SM = SingleNonDim(gp, ndd).run()
plotModel3(SM)
print(f'MI = {SM.mixIncrease}')

gp['mixAway'] = False
gp['q'] = 1/3000
ndd = makeNDDict(gp, Ra = SM.Ra, Fr = 1e-2, Fw = SM.Fw)
SM = SingleNonDim(gp, ndd).run()
#print(f'MI = {SM.mixIncrease}')



#print(f'Largest US:')
print(np.shape(SM.S))
print(f'max unstable: {np.amax(-SM.S[0,:]+SM.S[-1,:])}')

plt.figure()
plt.plot(SM.X, -SM.S[0,:]+SM.S[-1,:])
plt.plot(SM.X, SM.S[0,:])
plt.plot(SM.X, SM.S[-1,:])
plt.plot(SM.Sb)



# Results Section 3 Figs



dd, ndd = makeDicts(gp, 'tau_w', 'Q')
PS = ParameterSweep(gp,ndd,1).run()
plotDim(PS,dd)

# Shallow, low river discharge, tidally energetic
dd, ndd = makeDicts(gp, 'tau_w', Q = 200, K_H = 250, K_M = 1e-3, H = 3)
PS = ParameterSweep(gp,ndd,1).run()
plotDim(PS,dd)

# Deep, partially stratified
dd, ndd = makeDicts(gp, 'tau_w', Q = 5000, K_H = 1250, H = 70)
PS = ParameterSweep(gp,ndd,1).run()
plotDim(PS,dd)

# Varying H
dd, ndd = makeDicts(gp, 'H', tau_w = .2, K_M = 1e-2, K_H = 50)
PS = ParameterSweep(gp,ndd,1).run()
plotDim(PS,dd)

#Varying Q with subtidal wind present
dd, ndd = makeDicts(gp, 'Q', tau_w = -.1, K_H = 100, K_M = 5e-3, H = 50)
PS = ParameterSweep(gp,ndd,1).run()
plotDim(PS,dd)

#Varying KM with subtidal wind present
dd, ndd = makeDicts(gp, 'K_M', tau_w = .1, K_H = 10, H = 20)
PS = ParameterSweep(gp,ndd,1).run()
plotDim(PS,dd)
#dd, ndd = makeDicts(gp, 'Q', tau_w = -.1, K_H = 2, K_M = 1e-1, H = 60)
#PS = ParameterSweep(gp,ndd,1).run()
#plotDim(PS,dd)

#dd, ndd = makeDicts(gp, 'H', tau_w=-.2, Q = 200, K_H = 2, K_M = 1e-2)
#PS = ParameterSweep(gp,ndd,1).run()
#plotDim(PS,dd)


Est = [ 'Ch', 'Mh', 'Scheldt', 'Rotterdam', 'Sy', 'Hh',  'Du', 'Cl', 'Hudson (Low Q)', 'Ml', 'Dn', 'James']
FrD = [.3, .7,  9e-3, .2, .25, .15, .2, 4e-2, 1.5e-2, 7e-2, 2e-2, 5e-3]
RaD = [60, 80, 80, 200, 800, 850, 1000, 1800, 3000, 1e4, 1.3e4, 8e4]

selec = [2, 3, 8, 11]
PS = []
i = 0
for (fr, ra, name) in zip(FrD, RaD, Est):
    if i in selec:
        ndd = makeNDDict(gp, 'Fw', Fr = fr, Ra = ra, name = name)
        PS.append(ParameterSweep(gp, ndd, 0).run())
    i = i + 1
    #plotNDim(PS)
#plt.show()

plotNDimEst(PS)

#Paper: Draft 2

'''
"X" #No solution for Sb_X0, 
"/" #No solution for Xs, 
"+" #Non-unique solution (loc. extr.)
"."  #neg. sal
"O" #unstable strat
'''
PSList = []
'''
i = 0

for fr in [.0025, .025, .25, 1]:
        ndd = makeNDDict(gp, 'Fw', 'Ra', Fr = fr, name = 'fr')
        PSList.append(ParameterSweep(gp, ndd, 0).run())
        plotNDim(pp, PSList[i])
        i += 1
'''
#plotReg4(pp, PSList)
PSDList, dimDictList=[],[]
gp['mixAway'] = True
q = [500, 1300]
h = [30, 22]
kh = [1000, 150]
km = [1e-2, 5e-4]
twl = [[-3,3], [-.5, .5]]
for i in [0,1,2]:
        dd, ndd = makeDicts(gp, 'tau_w', Q = q[i], K_H = kh[i], K_M = km[i],H=h[i], tauwLim = twl[i])
        PSDList.append(ParameterSweep(gp, ndd, 1).run())
        dimDictList.append(dd)
#plotDim(pp,PS,dd)

gp['mixAway'] = False
for i in [0,1]:
        nddN= makeNDDict(gp, 'Fw', 'Ra', Fr = PSDList[i].Fr[0], name = 'fr')
        PSList.append(ParameterSweep(gp, nddN, 1).run())
plotDimNDim(pp, PSDList, dimDictList, PSList)
'''

Ravec = [1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4]
Fwvec =  [-1, -.25, 0, .25, 1, 2, 8]   #symlog10(np.linspace(invsymlog10(-10), invsymlog10(10), 11))
SM = [SingleNonDim(gp, makeNDDict(gp, Ra = ra, Fr = 1e-2, Fw = fw)).run() for ra,fw in zip(Ravec,Fwvec)]
plotCubics(pp,SM)


'''



# Figs Draft3

pp = plotParameters(hatches = False,
        mask = ~gp['mixAway'],
        We = False)


PSList = []

i = 0




for fr in [0.00025, 0.0025, 0.025, 0.1]:
        ndd = makeNDDict(gp, 'Fw', 'Ra', Fr = fr, name = 'fr')
        PSList.append(ParameterSweep(gp, ndd, 0).run())
        plotNDim(pp, PSList[i])
        i += 1

plotReg4(pp, PSList)



PSDList, PSList, dimDictList=[],[],[]
gp['mixAway'] = False
q = [1000, 1000, 1000]
h = [10, 18 , 50]
kh = [150, 150, 150]
km = [2e-2, 2e-2, 2e-2]
twl = [[-.5, 8], [-.5, 8], [-.5, 8]]
si = [0,1,2]
for i in si:
        dd, ndd = makeDicts(gp, 'tau_w', Q = q[i], K_H = kh[i], K_M = km[i],H=h[i], tauwLim = twl[i])
        PSDList.append(ParameterSweep(gp, ndd, 1).run())
        dimDictList.append(dd)
        #print(f"Regimes: H = {dimDictList[i]['H']}.")
#plotDim(pp,PS,dd)

gp['mixAway'] = False
for i in si:
        nddN = makeNDDict(gp, 'Fw', 'Ra', Fr = PSDList[i].Fr[0], name = 'fr')
        PSList.append(ParameterSweep(gp, nddN, 1).run())
        
        
plotDimNDim(pp, PSDList, dimDictList, PSList)


PSList = []

i = 0

PSDList, PSList, dimDictList=[],[],[]
gp['mixAway'] = False
q = [1000, 1000, 1000]
h = [10, 18 , 50]
kh = [150, 150, 150]
km = [2e-2, 2e-2, 2e-2]
twl = [[-.5, 8], [-.5, 8], [-.5, 8]]
si = [0,1,2]
for i in si:
        dd, ndd = makeDicts(gp, 'tau_w', Q = q[i], K_H = kh[i], K_M = km[i],H=h[i], tauwLim = twl[i], Sverdrup = 0.135/10, Guha = True)
        PSDList.append(ParameterSweep(gp, ndd, 1).run())
        dimDictList.append(dd)
        #print(f"Regimes: H = {dimDictList[i]['H']}.")
#plotDim(pp,PS,dd)

gp['mixAway'] = False
for i in si:
        nddN = makeNDDict(gp, 'Fw', 'Ra', Fr = PSDList[i].Fr[0], name = 'fr')
        PSList.append(ParameterSweep(gp, nddN, 1).run())
        
        
plotDimNDim(pp, PSDList, dimDictList, PSList)

Ras = [1000, 5e4]
Fws = [1.7, -.5]

for ra, fw in zip(Ras, Fws):
    ndd = makeNDDict(gp, Ra = ra, Fr = 0.025, Fw = fw)
    SM = SingleNonDim(gp, ndd).run()
    plotModel3(pp,SM)

plt.show()


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