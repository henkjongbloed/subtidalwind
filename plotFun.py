import numpy as np
import matplotlib.pyplot as plt
from numpy.core.arrayprint import array2string
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from matplotlib import ticker, cm
import matplotlib.tri as mtri
import matplotlib.colors as co
import matplotlib.cm as cm
from generalFun import *



def plotParameters(**kwargs):
    pp = dict(hatches = False)

    for key, value in kwargs.items(): #Change the constant parameters
            pp[key] = value
    return pp

def plotSpace(SM, PS, PSd):
    sRa, sFr, sFw = np.log10(PS.Ra.ravel()), np.log10(PS.Fr.ravel()), symlog10(PS.Fw.ravel())
    totMask = PS.totMask
    
    Reg =  np.array(PS.Reg.ravel())
    Reg[totMask] = 0
    #plt.plot(Reg)
    quan = [np.count_nonzero(np.round(Reg) == ind) for ind in range(5)]

    fig = plt.figure()
    fig.suptitle('3D Parameter Space (log axes)')
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    f1 = ax.scatter(sRa, sFr, sFw, c = PS.Reg.ravel(), cmap = plt.get_cmap('plasma'))
    cb = plt.colorbar(f1, ax = ax, ticks = [1 ,2, 3, 4])
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title('Regime: #' + str(quan))
    
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    f2 = ax.scatter(sRa, sFr, sFw, c = PS.Phi_0.ravel(), cmap = 'Blues')
    plt.colorbar(f2, ax = ax)
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title(r'Mouth stratification $\Phi_0$')
    #print(np.min(PS.Phi_0)) correct
    
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    f3 = ax.scatter(sRa, sFr, sFw, c = np.log10(PS.Ls.ravel()), cmap = 'Blues') #np.log10(PS.Ls.ravel())
    plt.colorbar(f3, ax = ax)
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title(r'Salt intrusion $L_s$ (log)')
    
    totMaskd = np.squeeze(PSd.totMask.reshape(PSd.nps))
    dRa, dFr, dFw = np.squeeze(np.log10(PSd.Ra.reshape(PSd.nps))),  np.squeeze(np.log10(PSd.Fr.reshape(PSd.nps))),  np.squeeze(symlog10(PSd.Fw.reshape(PSd.nps)))
    
    col = np.squeeze(PSd.Phi_0)
    #col[PSd.totMask] = 0
    
    #col = np.array(np.squeeze(col.reshape(PSd.nps)))
    norm = plt.Normalize(vmin=col.min(), vmax=col.max())
    Blues = plt.get_cmap('Blues')
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    f4 = ax.plot_surface(dRa, dFr, dFw, facecolors = Blues(norm(col)), cmap = 'Blues',  linewidth = 0, antialiased=True)#color = 
    plt.colorbar(f4, ax = ax)
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')    
    ax.set_title(r'Slice: $\tau_w - H$, color: $\Phi_0$, max = ' + np.array2string(np.amax(col)))
    
def plotDim(PS, dimDict):
    if len(dimDict['pars']) == 1:
        plotDim1(PS,dimDict)
    elif len(dimDict['pars']) == 2:
        plotDim2(PS,dimDict)
    elif len(dimDict['pars']) == 3:
        plotDim3(PS,dimDict)
        
def plotNDim(pp, PS):
    ndd = PS.nondimDict
    if len(ndd['pars']) == 1:
        plotNDim1(pp, PS)
    elif len(ndd['pars']) == 2:
        plotNDim2(pp, PS)
    elif len(ndd['pars']) == 3:
        plotNDim3(pp, PS)

def plotNDim1(pp,PS):
    col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    labT = ['G-G', 'G-R', 'G-W', 'R-R', 'R-W', 'W-W', 'DI', '|FL|']
    #namex = dimDict['pars'][0]
    #tm = np.transpose(np.array([PS.mask[:,5], PS.mask[:,5]], dtype = bool))
    # Prepare background color for plots
    ndd = PS.nondimDict
    namex = ndd['pars'][0]
    if PS.gp['Ori']:
        namex = namex+'Ori'

    if 'Fw' in namex:
        varx = symlog10(getattr(PS, namex).reshape(PS.nps))
    else:
        varx = np.log10(getattr(PS, namex).reshape(PS.nps))
    
    Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xs)), np.squeeze(PS.Reg)
    LsT = PS.LsT

    Regcolor = np.array([Reg, Reg])
    extentp = (np.amin(varx), np.amax(varx), np.amin(Phi_0), np.amax(Phi_0))
    Ll, Lu = 1/2*np.amin(Ls), 2*np.amax(Ls)
    extentl = (np.amin(varx), np.amax(varx), Ll, Lu)
    #x, y = x.ravel(), y.ravel()
    
    LD = np.ma.masked_outside(LsT[:,6], Ll, Lu)
    LGG = np.ma.masked_outside(LsT[:,0], Ll, Lu)
    LWW = np.ma.masked_outside(LsT[:,5], Ll, Lu)
    LGW = np.ma.masked_outside(LsT[:,-1], Ll, Lu)
    fig, axs = plt.subplots(3,1, sharex = True)

    #plt.tight_layout()

    plt.suptitle(PS.name)    
    f1 = axs[0].imshow(Regcolor, extent = extentp, origin = 'upper', aspect = 'auto')
    axs[0].plot(varx, Phi_0, lw = 2, c = 'k')
    axs[0].contourf(*np.meshgrid(varx, np.array([np.amin(Phi_0), np.amax(Phi_0)]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,0], PS.maskOri[:,0]])), 1, hatches = [" ", "X"], alpha = 0)
    axs[0].contourf(*np.meshgrid(varx, np.array([np.amin(Phi_0), np.amax(Phi_0)]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,2], PS.maskOri[:,2]])), 1, hatches = [" ", "+"], alpha = 0)
    axs[0].contourf(*np.meshgrid(varx, np.array([np.amin(Phi_0), np.amax(Phi_0)]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,3], PS.maskOri[:,3]])), 1, hatches = [" ", "."], alpha = 0)
    axs[0].contourf(*np.meshgrid(varx, np.array([np.amin(Phi_0), np.amax(Phi_0)]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,4], PS.maskOri[:,4]])), 1, hatches = [" ", "O"], alpha = 0)

    axs[0].title.set_text(r'Stratification $\Phi_0$')
    if 'Fw' in namex:
        axs[0].plot([0,0], np.array([np.amin(Phi_0), np.amax(Phi_0)]), c = 'w')

    axs[1].plot(varx, Ls, lw = 2, c = 'k', label = r'$L_s$')
    axs[1].title.set_text(r'Salt intrusion $\Lambda_s$ (NonDim)')
    f2 = axs[1].imshow(Regcolor, extent = extentl, origin = 'upper', aspect = 'auto')
    axs[1].plot(varx, LD, ls = '-', lw = 1, c = 'w', label = 'Dispersive') #Dispersive Regime
    axs[1].plot(varx, LGG, ls = '--', lw = 1, c = 'w', label = 'Chatwin') #Chatwin Regime
    axs[1].plot(varx, LWW, ls = '-.', lw = 1, c = 'w', label = 'Wind-driven') #WW Regime
    axs[1].plot(varx, LGW, ls = ':', lw = 1, c = 'w', label = 'GW-GG') #WW Regime
    axs[1].contourf(*np.meshgrid(varx, np.array([Ll, Lu]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,0], PS.maskOri[:,0]])), 1, hatches = [" ", "X"], alpha = 0)
    axs[1].contourf(*np.meshgrid(varx, np.array([Ll, Lu]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,2], PS.maskOri[:,2]])), 1, hatches = [" ", "+"], alpha = 0)
    axs[1].contourf(*np.meshgrid(varx, np.array([Ll, Lu]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,3], PS.maskOri[:,3]])), 1, hatches = [" ", "."], alpha = 0)
    axs[1].contourf(*np.meshgrid(varx, np.array([Ll, Lu]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,4], PS.maskOri[:,4]])), 1, hatches = [" ", "O"], alpha = 0)
    #print(LGW)
    if 'Fw' in namex:
        axs[1].plot([0,0], np.array([Ll, Lu]), c = 'w')
    axs[1].set_yscale('log')
    axs[1].legend()

    T = PS.T
    for ind in range(T.shape[1]-1): axs[2].plot(varx, np.squeeze(T[:,ind]), color=col[ind], label = labT[ind])
    ind = T.shape[1]-1
    axs[2].plot(varx, np.abs(np.squeeze(T[:,ind])), color = col[ind], label = labT[ind])
    axs[2].legend()
    axs[2].title.set_text('Transports')
    axs[2].set_xlabel(ndd['pars'][0])

def plotNDim2(pp,PS):
    ndd = PS.nondimDict
    namex, namey = ndd['pars']
    if PS.gp['Ori']:
        namex, namey = namex+'Ori', namey+'Ori'
    cmap = cm.get_cmap("Blues").copy()
    print(cmap.get_bad())
    cmap.set_bad(color = 'gray', alpha = .5)
    print(cmap.get_bad())

    #fig, axs = plt.subplots(3,1, sharex = True)
    nps = PS.nps
    if pp['mask']:
        Phi_0 = np.ma.masked_array(np.squeeze(PS.Phi_0), PS.mask[:,7])
        Ls = np.ma.masked_array(np.abs(np.squeeze(PS.Xs)), PS.mask[:,7])
        Reg = np.squeeze(PS.Reg)
        Regcolor = np.swapaxes(np.reshape(Reg, (*nps, 3)), 0, 1)
        #RegMask = np.reshape(PS.mask[:,7], PS.nps)
        #Regcolor = np.concatenate([Regcolor0, ~RegMask], axis=-1)
        Sb_X0 = np.ma.masked_array(np.squeeze(PS.Sb_X0), PS.mask[:,7])
        Ls0 = np.ma.masked_array(np.squeeze(np.abs(PS.Xs0)), PS.mask[:,7])
        Ls1 = np.ma.masked_array(np.squeeze(np.abs(PS.Xs1)), PS.mask[:,7])
        Sb_0 = np.ma.masked_array(np.squeeze(PS.Sb_0), PS.mask[:,7])
        Phi_c = np.ma.masked_array(np.squeeze(PS.Phi_c), PS.mask[:,7])
        Phi_s = np.ma.masked_array(np.squeeze(PS.Phi_s), PS.mask[:,7])
    else:
        Phi_0 = np.squeeze(PS.Phi_0)
        Ls = np.abs(np.squeeze(PS.Xs))
        Reg = np.squeeze(PS.Reg)
        Regcolor = np.swapaxes(np.reshape(Reg, (*nps, 3)), 0, 1)
        M0 = np.squeeze(PS.M0)
        Sb_X0 = np.squeeze(PS.Sb_X0)
        Sb_0 = np.squeeze(PS.Sb_0)
        Ls0 = np.abs(PS.Xs0.reshape(PS.nps))
        Ls1 = np.squeeze(np.abs(PS.Xs1))
        Phi_c = np.squeeze(PS.Phi_c)
        Phi_s = np.squeeze(PS.Phi_s)
    #PS.maxUN[PS.maxUN > 0] = 0
    maxUN = np.squeeze(PS.maxUN[:,0])
    mixFac = np.log10(np.squeeze(PS.mixIncrease))
    #print(mixFac)


    
    if 'Fw' in namex:
        if pp['We']:
            varx = symlog10(PS.We0.reshape(PS.nps))
            print(np.amax(varx))
            print(np.amin(varx))
        else:
            varx = symlog10(getattr(PS, namex).reshape(PS.nps))
    else:
        varx = np.log10(getattr(PS, namex).reshape(PS.nps))
    if 'Fw' in namey:
        vary = symlog10(getattr(PS, namey).reshape(PS.nps))
    else:
        vary = np.log10(getattr(PS, namey).reshape(PS.nps))
        
    fig = plt.figure(constrained_layout=True)
    #s3 = fig.add_gridspec(ncols=3, nrows=2)
    s3 = fig.add_gridspec(ncols=2, nrows=2)
    

    # We create a colormar from our list of colors
    #cm = co.ListedColormap(np.identity(3))
    #cm = co.LinearSegmentedColormap.from_list
    # Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
    labels = ['  III', '  IV']
    #len_lab = len(labels)
    
    # SM - locations in parameter space.
    Ras = [1000, 50000]
    Fws = [1.7, -.5]
    
    #if h[0]:
        #pass
    
    
    for a in range(4):
        ax = fig.add_subplot(s3[a])
        cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
        if a==0: 
            ax.set_title('Regime')
            #plt.colorbar(cm, ax=ax)
            if ndd['Fr'][0] == 0.025:
                Ras = np.log10(np.array(Ras))
                Fws = symlog10(np.array(Fws))
                ax.scatter(Fws, Ras, c = 'k')
                for i,lab in zip([0,1], labels):
                    ax.annotate(lab, (Fws[i], Ras[i]))
            #ax.set_ylabel(ndd['pars'][1])        
        elif a==2:
            #cfig = ax.contourf(varx, vary, np.log10(Ls.reshape(PS.nps)*Sb_X0.reshape(PS.nps)), 20, cmap = cmap)
            #cfig = ax.contourf(varx, vary, np.log10(Sb_X0.reshape(PS.nps)), 20, cmap = cmap)
            print(np.amin(Sb_X0))
            print(np.amax(Sb_X0))
            maxexp = int(-(-np.amax(np.log10(Sb_X0))//1))
            minexp = int(np.amin(np.log10(Sb_X0))//1)
            cfig = ax.contourf(varx, vary, np.log10(Sb_X0.reshape(PS.nps)), 20, cmap = cmap, corner_mask = True, vmin = minexp, vmax = maxexp)
            ax.set_title(r'Salinity gradient $\bar{\Sigma}_{X0}$')
            SbXTicks = np.arange(start = maxexp, stop = minexp-1, step = -1 ) #backward linspace
            cb = plt.colorbar(cfig, ax = ax, ticks = list(SbXTicks))
            
            
            SbXLab = ['$1$','$10^{-1}$','$10^{-2}$','$10^{-3}$','$10^{-4}$','$10^{-5}$','$10^{-6}$','$10^{-7}$']
            cb.set_ticklabels(SbXLab[-maxexp+1:-minexp+2])
            #cb = plt.colorbar(cfig, ax = ax)
        elif a==1:
            cfig = ax.contourf(varx, vary, Phi_0.reshape(PS.nps), 20, cmap = cmap, corner_mask = True)
            ax.set_title(r'Stratification $\Phi_0$')
            cb = plt.colorbar(cfig, ax = ax)
            #cb.set_ticklabels(f'{l:1.2f}' for l in cb.get_ticks())
        elif a==3:
            print(np.amin(Ls))
            print(np.amax(Ls))
            maxexp = int(-(-np.amax(np.log10(Ls))//1))
            minexp = int(np.amin(np.log10(Ls))//1)
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), np.log10(Ls.reshape(PS.nps)), 20, cmap = cmap, corner_mask = True, vmin = minexp, vmax = maxexp)
            ax.set_title(r'Salt intrusion $\Lambda_s$')
            LsLab = ['$1$','$10$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$', '$10^7$', '$10^8$', '$10^9$', '$10^{10}$', '$10^{11}$']
            LsTicks =  list(range(minexp, maxexp+1))
            cb = plt.colorbar(cfig, ax = ax, ticks = LsTicks)
            cb.set_ticklabels(LsLab[minexp:(maxexp+1)])

        if pp['hatches']:
            #if a==0:
            ax.contourf(varx, vary, PS.maskOri[:,0].reshape(PS.nps), 1, hatches =  [None,"X"], alpha = 0)
            ax.contourf(varx, vary, PS.maskOri[:,1].reshape(PS.nps), 1, hatches =  [None,"/"], alpha = 0)
            ax.contourf(varx, vary, PS.maskOri[:,2].reshape(PS.nps), 1, hatches =  [None,"+"], alpha = 0)
            ax.contourf(varx, vary, PS.maskOri[:,3].reshape(PS.nps), 1, hatches =  [None,"."], alpha = 0)
            ax.contourf(varx, vary, PS.maskOri[:,4].reshape(PS.nps), 1, hatches =  [None,"O"], alpha = 0) 
        #plt.yticks([-4,-3,-2,-1,0,1,2,3,4,5,6], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])
        #if a not in [3,4,5]:
            
        
        if a in [0,1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(ndd['pars'][0])

        if a in [1,3]:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel(ndd['pars'][1])


        if 'Fw' in namex:
            #print(invsymlog10(np.amax(varx)))
            numTicks = 7
            ax.plot([0,0], [np.amin(vary), np.amax(vary)], c = 'w', lw = .5, ls = ':') #look at this!!!
            FwVal = invsymlog10(np.linspace(np.amin(varx), np.amax(varx), numTicks))
            FwTicks = []
            for t in range(numTicks): FwTicks.append(f'{FwVal[t]:1.1f}')
            plt.xticks(np.linspace(np.amin(varx), np.amax(varx), numTicks), FwTicks)
            #print(invsymlog10(np.linspace(np.amin(varx), np.amax(varx), 11)))
            #print(type(invsymlog10(np.linspace(np.amin(varx), np.amax(varx), 11))))
            #print(np.linspace(symlog10(np.amin(varx)), symlog10(np.amax(varx)), 11))
            #print(np.amax(varx))
        elif 'Ra' in namex:
            plt.xticks([2,3,4,5], ['$10^2$', '$10^3$', '$10^4$', '$10^5$'])
        elif 'Fr' in namey:
            plt.xticks([-4,-3,-2,-1,0], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
        if 'Fw' in namey:
            ax.plot([np.amin(varx), np.amax(varx)],[0,0], c = 'w', lw = .5) #look at this!!!
            plt.yticks(np.linspace(np.amin(varx), np.amax(varx), 11), ['$-10^2$','$-10^1$', '$-1$', '$-10^{-1}$', '$-10^{-2}$', '$0$','$10^{-2}$', '$10^{-1}$', '$1$', '$10^1$', '$10^2$'])
            ax.set_yscale('symlog')
            #print(np.linspace(symlog10(np.amin(varx)), symlog10(np.amax(varx)), 11))
            #print(np.amax(varx))
        elif 'Ra' in namey:
            plt.yticks([2,3,4,5], ['$10^2$', '$10^3$', '$10^4$', '$10^5$'])
        elif 'Fr' in namey:
            plt.yticks([-4,-3,-2,-1,0], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])



def plotNDim22(pp,PS):
    ndd = PS.nondimDict
    namex, namey = ndd['pars']
    if PS.gp['Ori']:
        namex, namey = namex+'Ori', namey+'Ori'
    cmap = cm.get_cmap("Blues").copy()
    print(cmap.get_bad())
    cmap.set_bad(color = 'gray', alpha = .5)
    print(cmap.get_bad())

    #fig, axs = plt.subplots(3,1, sharex = True)
    nps = PS.nps
    if pp['mask']:
        Phi_0 = np.ma.masked_array(np.squeeze(PS.Phi_0), PS.mask[:,7])
        Phi_c = np.ma.masked_array(np.squeeze(PS.Phi_c), PS.mask[:,7])
        Phi_s = np.ma.masked_array(np.squeeze(PS.Phi_s), PS.mask[:,7])
        Ls = np.ma.masked_array(np.abs(np.squeeze(PS.Xs)), PS.mask[:,7])
        Ls0 = np.ma.masked_array(np.abs(np.squeeze(PS.Xs0)), PS.mask[:,7])
        Ls1 = np.ma.masked_array(np.abs(np.squeeze(PS.Xs1)), PS.mask[:,7])

        Reg = np.squeeze(PS.Reg)
        Regcolor = np.swapaxes(np.reshape(Reg, (*nps, 3)), 0, 1)
        #RegMask = np.reshape(PS.mask[:,7], PS.nps)
        #Regcolor = np.concatenate([Regcolor0, ~RegMask], axis=-1)
        M0 = np.ma.masked_array(np.squeeze(PS.M0), PS.mask[:,7])
        Sb_X0 = np.ma.masked_array(np.squeeze(PS.Sb_X0), PS.mask[:,7])
        Sb_0 = np.ma.masked_array(np.squeeze(PS.Sb_0), PS.mask[:,7])
        Sb_X02 = np.ma.masked_array(Sb_0/Ls*(1-PS.gp['ep']), PS.mask[:,7])

    else:
        Phi_0 = np.squeeze(PS.Phi_0)
        Ls = np.abs(np.squeeze(PS.Xs))
        Reg = np.squeeze(PS.Reg)
        Regcolor = np.swapaxes(np.reshape(Reg, (*nps, 3)), 0, 1)
        M0 = np.squeeze(PS.M0)
        Sb_X0 = np.squeeze(PS.Sb_X0)
        Sb_0 = np.squeeze(PS.Sb_0)
        Sb_X02 = Sb_0/Ls*(1-PS.gp['ep'])
    #PS.maxUN[PS.maxUN > 0] = 0
    maxUN = np.squeeze(PS.maxUN[:,0])
    mixFac = np.log10(np.squeeze(PS.mixIncrease))
    #print(mixFac)


    
    if 'Fw' in namex:
        if pp['We']:
            varx = symlog10(PS.We0.reshape(PS.nps))
            print(np.amax(varx))
            print(np.amin(varx))
        else:
            varx = symlog10(getattr(PS, namex).reshape(PS.nps))
    else:
        varx = np.log10(getattr(PS, namex).reshape(PS.nps))
    if 'Fw' in namey:
        vary = symlog10(getattr(PS, namey).reshape(PS.nps))
    else:
        vary = np.log10(getattr(PS, namey).reshape(PS.nps))
        
    fig = plt.figure(constrained_layout=True)
    #s3 = fig.add_gridspec(ncols=3, nrows=2)
    s3 = fig.add_gridspec(ncols=4, nrows=3)
    

    # We create a colormar from our list of colors
    #cm = co.ListedColormap(np.identity(3))
    #cm = co.LinearSegmentedColormap.from_list
    # Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
    labels = ['  I', '  II', '  III', '  IV']
    #len_lab = len(labels)
    
    # SM - locations in parameter space.
    Ras = [1000, 1e5, 1e4, 5e5, 1e3]
    Fws = [-1, .03, 1.7, -2, .3]
    
    #if h[0]:
        #pass
    
    
    for a in range(12):
        ax = fig.add_subplot(s3[a])
        if a==0: 
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            ax.set_title('Regime')
        elif a==1:
            #cfig = ax.contourf(varx, vary, Phi_0.reshape(PS.nps), 20, cmap = cmap, corner_mask = True, vmin = np.min(Phi_0), vmax = np.max(Phi_0))
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            #cfig = ax.contourf(varx, vary, PS.mask[:,7].reshape(PS.nps), 20, cmap = cmap, corner_mask = False)
            cfig = ax.contourf(varx, vary, Phi_0.reshape(PS.nps), 20, cmap = cmap, corner_mask = True)
            ax.set_title(r'Stratification $\Phi_0$')
            cb = plt.colorbar(cfig, ax = ax)
            #cb.set_ticklabels(f'{l:1.2f}' for l in cb.get_ticks())
        elif a==2:
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            #cfig = ax.contourf(varx, vary, M0.reshape(PS.nps), 20, cmap = cmap, vmin = np.min(M0), vmax = np.max(M0))
            cfig = ax.contourf(varx, vary, np.log10(np.abs(np.squeeze(PS.Xs))).reshape(PS.nps), 20, cmap = cmap)
            #ax.contour(cfig, levels = np.array([-1,0,1]), colors = 'k', linewidths = .5)
            ax.set_title(r'Surface SI')
            
            
        elif a==3:
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            #cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Ls.reshape(PS.nps), 20, cmap = cmap, corner_mask = True)
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), np.log10(Ls.reshape(PS.nps)), 20, cmap = cmap)
            ax.set_title(r'Salt intrusion $\Lambda_s$')
            #cb = plt.colorbar(cfig, ax = ax)
            
            maxexp = int(-(-np.amax(np.log10(Ls))//1))
            minexp = int(np.amin(np.log10(Ls))//1)
            LsLab = ['$1$','$10$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$', '$10^7$', '$10^8$', '$10^9$', '$10^{10}$', '$10^{11}$']
            LsTicks =  list(range(minexp, maxexp+1))
            cb = plt.colorbar(cfig, ax = ax, ticks = LsTicks)
            cb.set_ticklabels(LsLab[minexp:(maxexp+1)])
            #ax.set_xlabel(ndd['pars'][0])

        elif a==4:
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            #cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Ls.reshape(PS.nps), 20, cmap = cmap, corner_mask = True)
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Sb_0.reshape(PS.nps), 20, cmap = cmap)
            ax.set_title(r'Sal0')
            cb = plt.colorbar(cfig, ax = ax)
            
            
        elif a==5:
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            #cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Ls.reshape(PS.nps), 20, cmap = cmap, corner_mask = True)
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), np.log10(Sb_X0.reshape(PS.nps)), 20, cmap = cmap)
            ax.set_title(r'Sal Gradient')
            cb = plt.colorbar(cfig, ax = ax)
            
        elif a==6:
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            #cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Ls.reshape(PS.nps), 20, cmap = cmap, corner_mask = True)
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), np.log10(Sb_X02.reshape(PS.nps)), 20, cmap = cmap)
            ax.set_title(r'Sal Grad 2')
            cb = plt.colorbar(cfig, ax = ax)

        elif a==7:
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            #cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Ls.reshape(PS.nps), 20, cmap = cmap, corner_mask = True)
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), np.log10(np.abs((Sb_X0.reshape(PS.nps)-Sb_X02.reshape(PS.nps))/Sb_X0.reshape(PS.nps))), 20, cmap = cmap)
            ax.set_title(r'Sal Grad Diff')
            cb = plt.colorbar(cfig, ax = ax)

        elif a==8:
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            #cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Ls.reshape(PS.nps), 20, cmap = cmap, corner_mask = True)
            #np.ma.masked_array(np.squeeze(PS.Sb_0), PS.mask[:,7])
            Sb_X0I = np.ma.masked_array(PS.Fr*PS.Sb_0, PS.mask[:,7])
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), np.log10(Sb_X0I.reshape(PS.nps)), 20, cmap = cmap)
            ax.set_title(r'Sal Grad I')
            cb = plt.colorbar(cfig, ax = ax)
            
        elif a==9:
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            Sb_X0II = np.ma.masked_array(PS.Fr*PS.Sb_0, PS.mask[:,7])
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), np.log10(Sb_X0II.reshape(PS.nps)), 20, cmap = cmap)
            ax.set_title(r'Sal Grad II')
            cb = plt.colorbar(cfig, ax = ax)
            
        elif a==10:
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            Sb_X0III = np.ma.masked_array(PS.Fr/PS.L[:,5]*PS.Sb_0, PS.mask[:,7])
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), np.log10(Sb_X0III.reshape(PS.nps)), 20, cmap = cmap)
            ax.set_title(r'Sal Grad III')
            cb = plt.colorbar(cfig, ax = ax)
            
        elif a==11:
            cfig = ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            Sb_X0IV = np.ma.masked_array(np.abs(PS.L[:,2]/PS.L[:,0]), PS.mask[:,7])
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), np.log10(Sb_X0IV.reshape(PS.nps)), 20, cmap = cmap)
            ax.set_title(r'Sal Grad IV')
            cb = plt.colorbar(cfig, ax = ax)

        if pp['hatches']:
            #if a==0:
            ax.contourf(varx, vary, PS.maskOri[:,0].reshape(PS.nps), 1, hatches =  [None,"X"], alpha = 0)
            ax.contourf(varx, vary, PS.maskOri[:,1].reshape(PS.nps), 1, hatches =  [None,"/"], alpha = 0)
            ax.contourf(varx, vary, PS.maskOri[:,2].reshape(PS.nps), 1, hatches =  [None,"+"], alpha = 0)
            ax.contourf(varx, vary, PS.maskOri[:,3].reshape(PS.nps), 1, hatches =  [None,"."], alpha = 0)
            ax.contourf(varx, vary, PS.maskOri[:,4].reshape(PS.nps), 1, hatches =  [None,"O"], alpha = 0) 
        #plt.yticks([-4,-3,-2,-1,0,1,2,3,4,5,6], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])
        #if a not in [3,4,5]:
            
        
        if a in [0,1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(ndd['pars'][0])

        if a in [1,3]:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel(ndd['pars'][1])


        if 'Fw' in namex:
            #print(invsymlog10(np.amax(varx)))
            numTicks = 7
            #ax.plot([0,0], [np.amin(vary), np.amax(vary)], c = 'w', lw = .5, ls = ':') #look at this!!!
            FwVal = invsymlog10(np.linspace(np.amin(varx), np.amax(varx), numTicks))
            FwTicks = []
            #for t in range(numTicks): FwTicks.append(f'{FwVal[t]:1.1f}')
            #plt.xticks(np.linspace(np.amin(varx), np.amax(varx), numTicks), FwTicks)
            #print(invsymlog10(np.linspace(np.amin(varx), np.amax(varx), 11)))
            #print(type(invsymlog10(np.linspace(np.amin(varx), np.amax(varx), 11))))
            #print(np.linspace(symlog10(np.amin(varx)), symlog10(np.amax(varx)), 11))
            #print(np.amax(varx))
        elif 'Ra' in namex:
            plt.xticks([2,3,4,5], ['$10^2$', '$10^3$', '$10^4$', '$10^5$'])
        elif 'Fr' in namey:
            plt.xticks([-4,-3,-2,-1,0], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
        if 'Fw' in namey:
            ax.plot([np.amin(varx), np.amax(varx)],[0,0], c = 'w', lw = .5) #look at this!!!
            plt.yticks(np.linspace(np.amin(varx), np.amax(varx), 11), ['$-10^2$','$-10^1$', '$-1$', '$-10^{-1}$', '$-10^{-2}$', '$0$','$10^{-2}$', '$10^{-1}$', '$1$', '$10^1$', '$10^2$'])
            ax.set_yscale('symlog')
            #print(np.linspace(symlog10(np.amin(varx)), symlog10(np.amax(varx)), 11))
            #print(np.amax(varx))
        #elif 'Ra' in namey:
            #plt.yticks([2,3,4,5], ['$10^2$', '$10^3$', '$10^4$', '$10^5$'])
        elif 'Fr' in namey:
            plt.yticks([-4,-3,-2,-1,0], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])



def plotNDim2Data(pp,PS):
    ndd = PS.nondimDict
    namex, namey = ndd['pars']
    if PS.gp['Ori']:
        namex, namey = namex+'Ori', namey+'Ori'
    cmap = 'Blues'
    #fig, axs = plt.subplots(3,1, sharex = True)
    Phi_0 = np.squeeze(PS.Phi_0)
    Ls = np.abs(np.squeeze(PS.Xs))
    Reg = np.squeeze(PS.Reg)
    M0 = np.squeeze(PS.M0)
    #PS.maxUN[PS.maxUN > 0] = 0
    maxUN = np.squeeze(PS.maxUN[:,0])
    mixFac = np.log10(np.squeeze(PS.mixIncrease))
    #print(mixFac)
    nps = PS.nps

    Regcolor = np.swapaxes(np.reshape(Reg, (*nps, 3)), 0, 1)
    
    if 'Fw' in namex:
        varx = symlog10(getattr(PS, namex).reshape(PS.nps))
    else:
        varx = np.log10(getattr(PS, namex).reshape(PS.nps))
    if 'Fw' in namey:
        vary = symlog10(getattr(PS, namey).reshape(PS.nps))
    else:
        vary = np.log10(getattr(PS, namey).reshape(PS.nps))
        
    fig = plt.figure(constrained_layout=True)
    #s3 = fig.add_gridspec(ncols=3, nrows=2)
    s3 = fig.add_gridspec(ncols=2, nrows=2)
    

    # We create a colormar from our list of colors
    #cm = co.ListedColormap(np.identity(3))
    #cm = co.LinearSegmentedColormap.from_list
    # Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
    labels = ['I', 'II', 'III']
    #len_lab = len(labels)
    
    # SM - locations in parameter space.
    Ras = [100, 1e5, 1e4, 5e5, 1e3]
    Fws = [0.17, .03, 1.7, -2, .3]
    
    hatches = False
    for a in range(4):
        ax = fig.add_subplot(s3[a])
        if a==0: 
            cfig=ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            ax.set_title('Regime')
            #plt.colorbar(cm, ax=ax)
            if ndd['Fr'][0] == 1e-2:
                Ras = [1000, 1e5, 1e4, 5e5, 1e3]
                Fws = [-1, .03, 1.7, -2, .3]
                ax.scatter(symlog10(np.array(Fws)), np.log10(np.array(Ras)), c = 'k')
            ax.set_ylabel(ndd['pars'][1])
        elif a==1:
            cfig = ax.contourf(varx, vary, Phi_0.reshape(PS.nps), 20, cmap = cmap, corner_mask = True, vmin = np.min(Phi_0), vmax = np.max(Phi_0))
            ax.set_title(r'Stratification $\Phi_0$')
            plt.colorbar(cfig, ax = ax)
        elif a==2:
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), np.log10(Ls.reshape(PS.nps)), 20, cmap = cmap, corner_mask = True)
            ax.set_title(r'Salt intrusion $\Lambda_s$')
            plt.colorbar(cfig, ax = ax)
            #ax.set_xlabel(ndd['pars'][0])
        elif a==3:
            cfig = ax.contourf(varx, vary, M0.reshape(PS.nps), 20, cmap = cmap, corner_mask = True, vmin = np.min(M0), vmax = np.max(M0))
            ax.contour(cfig, levels = symlog10(np.array([-1,0,1])), colors = 'k', linewidths = .5)
            ax.set_title(r'Circulation parameter $M_0$')

            ax.set_xlabel(ndd['pars'][0])
            ax.set_ylabel(ndd['pars'][1])
        elif a==4:
            cfig = ax.contourf(varx, vary, maxUN.reshape(PS.nps), 20, cmap = cmap, corner_mask = True, vmin = np.min(maxUN), vmax = np.max(maxUN))
            #ax.contour(cfig, levels = symlog10(np.array([-1,0,1])), colors = 'k', lw = .5)
            ax.set_title(r'Largest unstable stratification')
            cb = plt.colorbar(cfig, ax = ax)
            #cb.ax.set_xtickslabels()            
            #ax.set_ylabel(ndd['pars'][1])
            ax.set_xlabel(ndd['pars'][0])
        elif a==5:
            cfig = ax.contourf(varx, vary, mixFac.reshape(PS.nps), 20, cmap = cmap, corner_mask = True, vmin = np.min(mixFac), vmax = np.max(mixFac))
            ax.contour(cfig, levels = [0,1,2,3,4,5], colors = 'k', linewidths = .5)
            ax.set_title(r'Mixing increase factor')
            cb = plt.colorbar(cfig, ax = ax)
            #cb.ax.set_xtickslabels()
            ax.set_xlabel(ndd['pars'][0])
            
        if hatches:
            if a==0:
                ax.contourf(varx, vary, PS.maskOri[:,0].reshape(PS.nps), 1, hatches = [" ", "X"], alpha = 0)
                ax.contourf(varx, vary, PS.maskOri[:,2].reshape(PS.nps), 1, hatches = [" ", "+"], alpha = 0)
                ax.contourf(varx, vary, PS.maskOri[:,3].reshape(PS.nps), 1, hatches = [" ", "."], alpha = 0)
                ax.contourf(varx, vary, PS.maskOri[:,4].reshape(PS.nps), 1, hatches = [" ", "O"], alpha = 0) 
        #plt.yticks([-4,-3,-2,-1,0,1,2,3,4,5,6], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])
        #if a not in [3,4,5]:
            #plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel(ndd['pars'][0])
        if a not in [0, 3]:
            plt.setp(ax.get_yticklabels(), visible=False)


        if 'Fw' in namex:
            #print(invsymlog10(np.amax(varx)))
            ax.plot([0,0], [np.amin(vary), np.amax(vary)], c = 'k', lw = .5) #look at this!!!
            FwVal = invsymlog10(np.linspace(np.amin(varx), np.amax(varx), 11))
            FwTicks = []
            for t in range(11): FwTicks.append(f'{FwVal[t]:.2n}')
            plt.xticks(np.linspace(np.amin(varx), np.amax(varx), 11), FwTicks)
            #print(invsymlog10(np.linspace(np.amin(varx), np.amax(varx), 11)))
            #print(type(invsymlog10(np.linspace(np.amin(varx), np.amax(varx), 11))))
            #print(np.linspace(symlog10(np.amin(varx)), symlog10(np.amax(varx)), 11))
            #print(np.amax(varx))
        elif 'Ra' in namex:
            plt.xticks([1,2,3,4,5,6], ['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])
        elif 'Fr' in namey:
            plt.xticks([-4,-3,-2,-1,0], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
        if 'Fw' in namey:
            ax.plot([np.amin(varx), np.amax(varx)],[0,0], c = 'w', lw = .5) #look at this!!!
            plt.yticks(np.linspace(np.amin(varx), np.amax(varx), 11), ['$-10^2$','$-10^1$', '$-1$', '$-10^{-1}$', '$-10^{-2}$', '$0$','$10^{-2}$', '$10^{-1}$', '$1$', '$10^1$', '$10^2$'])
            ax.set_yscale('symlog')
            #print(np.linspace(symlog10(np.amin(varx)), symlog10(np.amax(varx)), 11))
            #print(np.amax(varx))
        elif 'Ra' in namey:
            plt.yticks([1,2,3,4,5,6], ['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])
        elif 'Fr' in namey:
            plt.yticks([-4,-3,-2,-1,0], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])

def plotDim1(pp, PS, dimDict):
    col = ['tab:green','tab:green', 'tab:orange', 'tab:green','tab:orange','tab:orange', 'tab:blue', 'tab:gray']
    ls = ['-', '--', '--', '-.', '-.', '-', '-', '-']    
    labT = ['GG', 'GR', 'GW', 'RR', 'RW', 'WW', 'D', '|FL|']
    namex = dimDict['pars'][0]
    #tm = np.transpose(np.array([PS.mask[:,5], PS.mask[:,5]], dtype = bool))
    # Prepare background color for plots
    
    #Reg = ma.masked_array(Reg, mask = tm)
    
    LsT = PS.LsT
    
    fig = plt.figure(constrained_layout=True)
    s4 = fig.add_gridspec(ncols = 1, nrows = 3) 
    #plt.suptitle(dimDict['name'][0])
    #plt.tight_layout()

    Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xs)), np.squeeze(PS.Reg)
    LDim = dimDict['K_H']/dimDict['c']
    LsDim = Ls*LDim

    if namex == 'tau_w':
        varx = symlog10(dimDict[namex])
    else:
        varx = np.log10(dimDict[namex])

    Regcolor = np.array([Reg, Reg])
    extentp = (np.amin(varx), np.amax(varx), np.amin(Phi_0), np.amax(Phi_0))
    #print(extentp)
    Ll, Lu = np.amin(Ls)/1.25, 1.25*np.amax(Ls)
    extentl = (np.amin(varx), np.amax(varx), Ll, Lu)
    LlDim, LuDim = np.amin(Ll*LDim)/1.25, np.amax(Lu*LDim)*1.25
    extentld = (np.amin(varx), np.amax(varx), LlDim, LuDim)
    #x, y = x.ravel(), y.ravel()
    
    LD = np.ma.masked_outside(LsT[:,6], Ll, Lu)
    LGG = np.ma.masked_outside(LsT[:,0], Ll, Lu)
    LWW = np.ma.masked_outside(LsT[:,5], Ll, Lu)
    LGW = np.ma.masked_outside(LsT[:,-1], Ll, Lu)
    LsDim = np.ma.masked_outside(LsDim, LlDim, LuDim) 
    LDd = LD*LDim
    LGGd =  LGG*LDim
    LWWd = LWW*LDim
    LGWd = LGW*LDim
    
    for a in range(3):
        ax = fig.add_subplot(s4[a])
        if a==0:
            ax.plot(varx, Phi_0, lw = 2, c = 'k')
            ax.title.set_text(r'Stratification $\Phi_0$')
            f1 = ax.imshow(Regcolor, extent = extentp, origin = 'upper', aspect = 'auto')
        #axs[0].contourf(*np.meshgrid(varx, np.array([np.amin(Phi_0), np.amax(Phi_0)]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,0], PS.maskOri[:,0]])), 1, hatches = [" ", "X"], alpha = 0)
        #axs[0].contourf(*np.meshgrid(varx, np.array([np.amin(Phi_0), np.amax(Phi_0)]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,2], PS.maskOri[:,2]])), 1, hatches = [" ", "+"], alpha = 0)
        #axs[0].contourf(*np.meshgrid(varx, np.array([np.amin(Phi_0), np.amax(Phi_0)]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,3], PS.maskOri[:,3]])), 1, hatches = [" ", "."], alpha = 0)
        #axs[0].contourf(*np.meshgrid(varx, np.array([np.amin(Phi_0), np.amax(Phi_0)]), indexing = 'ij'), np.transpose(np.array([PS.maskOri[:,4], PS.maskOri[:,4]])), 1, hatches = [" ", "O"], alpha = 0)

        #if 'tau_w' in namex:
            #ax.plot([0,0], np.array([np.amin(Phi_0), np.amax(Phi_0)]), c = 'w')
        #cb = plt.colorbar(f1, ax = axs[0], ticks = [1 ,2, 3, 4])
        #axs[0].set_yscale('log')
        #if a==1:
        #    ax.plot(varx, Ls, lw = 2, c = 'k', label = r'$L_s$')
        #    ax.title.set_text(r'Salt intrusion $\Lambda_s$ (NonDim)')
         #   f2 = ax.imshow(Regcolor, extent = extentl, origin = 'upper', aspect = 'auto')
        #    ax.plot(varx, LD, ls = '-', lw = 1, c = 'w', label = 'Dispersive') #Dispersive Regime
         #   ax.plot(varx, LGG, ls = '--', lw = 1, c = 'w', label = 'Chatwin') #Chatwin Regime
         #   ax.plot(varx, LWW, ls = '-.', lw = 1, c = 'w', label = 'Wind-driven') #WW Regime
         #   ax.plot(varx, LGW, ls = ':', lw = 1, c = 'w', label = 'GG-GW') #WW Regime

         #   ax.legend()
         #   ax.set_yscale('log')
        #if 'tau_w' in namex:
            #ax.plot([0,0], np.array([np.amin(Ls), np.amax(Ls)]), c = 'w')
        
        if a==1:            
            ax.plot(varx, LsDim, lw = 4, c = 'k', label = r'$L_s$')
            ax.plot(varx, LDd, ls = '-', lw = 1, c = 'w', label = 'Dispersive') #Dispersive Regime
            ax.plot(varx, LGGd, ls = '--', lw = 1, c = 'w', label = 'Chatwin') #Chatwin Regime
            ax.plot(varx, LWWd, ls = '-.', lw = 1, c = 'w', label = 'Wind-driven') #WW Regime
            ax.plot(varx, LGWd, ls = ':', lw = 1, c = 'w', label = 'GG-GW') #WW Regime
            ax.title.set_text(r'Salt intrusion $L_s$')
            ax.imshow(Regcolor, extent = extentld, origin = 'upper', aspect = 'auto')

            ax.set_yscale('log')
            #ax.set_ylabel(r'$L_s$')


        #ax.set_yscale('log')
        #if 'tau_w' in namex:
            #ax.plot([0,0], np.array([np.amin(LsDim), np.amax(LsDim)]), c = 'w')
            #ax.legend()
        # Here, insert theoretical prediction.
        if a==2:
            T = PS.T
            for ind in range(T.shape[1]-1): ax.plot(varx, np.squeeze(T[:,ind]), color=col[ind], ls = ls[ind], label = labT[ind])
            ind = T.shape[1]-1
            ax.plot(varx, np.abs(np.squeeze(T[:,ind])), color = col[ind], ls = ls[ind], label = labT[ind])
            #ax.legend()
            ax.title.set_text('Transport mechanisms')
            #ax.set_yscale('log')
            
        if 'tau_w' in namex:
            FwVal = invsymlog10(np.linspace(np.amin(varx), np.amax(varx), 11))
            FwTicks = []
            for t in range(11): FwTicks.append(f'{FwVal[t]:.2n}')
            plt.xticks(np.linspace(np.amin(varx), np.amax(varx), 11), FwTicks)
            xl = r'$\tau_w$'
        elif 'Q' in namex:
            plt.xticks([1,2,3,4], ['$10^1$', '$10^2$', '$10^3$', '$10^4$'])
            xl = r'$Q$'
        elif 'H' in namex:
            plt.xticks([0,1,2], ['$1$', '$10^1$', '$10^2$'])
            xl = r'$H$'
        elif 'K_M' in namex:
            plt.xticks([-4,-3,-2,-1,0], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
            xl = r'$K_M$'
        elif 'K_H' in namex:
            plt.xticks([0,1,2,3,4,5], ['$1$','$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$'])
            xl = r'$K_H$'
            
        if a in [0,1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        if a==2:
            ax.set_xlabel(xl)

    #for i in range(4):
        #if namex == 'tau_w':
            #axs[i].set_xscale('linear')
        #else:
            #axs[i].set_xscale('linear')
        #axs[i].set_xlabel(namex)
        #axs[i].grid(True)

    
def plotDim2(pp,PS, dimDict):
    namex, namey = dimDict['pars']
    cmap = 'Blues'
    fig = plt.figure(constrained_layout=True)
    s4 = fig.add_gridspec(ncols = 2, nrows = 2)    
    Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xs)), np.squeeze(PS.Reg)
    nps = PS.nps
    if 'tau_w' in namex:
        varx = symlog10(dimDict[namex].reshape(PS.nps))
    else:
        varx = np.log10(dimDict[namex].reshape(PS.nps))
        
    Regcolor = np.reshape(Reg, (*nps, 3))
    
    if 'tau_w' in namey:
        vary = symlog10(dimDict[namey].reshape(PS.nps))
    else:
        vary = np.log10(dimDict[namey].reshape(PS.nps))
        
    LDim = dimDict['K_H']/dimDict['c']*np.ones_like(Ls)
    LsDim = Ls*LDim
    #vary = np.log10(dimDict[namey].reshape(PS.nps))
    '''
    for i in range(4):
        if 'tau_w' in namey:
            axs[i].plot([0,0], [np.amin(vary), np.amax(vary)], c = 'w') 
        #else:
            #axs[i].set_xlabel(namex)
            
        if 'tau_w' in namey:
            axs[i].set_ylabel(r'$\tau_w$')
        else:
            axs[i].set_ylabel(dimDict['pars'][1]) 
    '''
    M0 = symlog10(np.squeeze(PS.M0))

    hatches = False
    for a in range(4):
        ax = fig.add_subplot(s4[a])
        if a==0: 
            cfig=ax.imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
            ax.set_title('Regime')
            #plt.colorbar(cm, ax=ax)
        elif a==1:
            cfig = ax.contourf(varx, vary, Phi_0.reshape(PS.nps), 20, cmap = cmap, corner_mask = True, vmin = np.min(Phi_0), vmax = np.max(Phi_0))
            ax.set_title(r'Stratification $\Phi_0$')
            plt.colorbar(cfig, ax = ax)
        elif a==2:
            cfig = ax.contourf(varx, vary, M0.reshape(PS.nps), 20, cmap = cmap, vmin = np.min(M0), vmax = np.max(M0))
            ax.contour(cfig, levels = symlog10(np.array([-1,0,1])), colors = 'k', linewidths = .5)
            ax.set_title(r'Circulation parameter $M_0$')
            #plt.colorbar(cfig, ax = ax)
        elif a==3:
            cfig = ax.contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), LsDim.reshape(PS.nps), 50, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True)
            ax.set_title(r'Salt intrusion $L_s$')
            plt.colorbar(cfig, ax = ax)
        if hatches and a==0:
            ax.contourf(varx, vary, PS.maskOri[:,0].reshape(PS.nps),1, hatches = [" ", "X"], alpha = 0)
            ax.contourf(varx, vary, PS.maskOri[:,2].reshape(PS.nps), 1, hatches = [" ", "+"], alpha = 0)
            ax.contourf(varx, vary, PS.maskOri[:,3].reshape(PS.nps), 1, hatches = [" ", "."], alpha = 0)
            ax.contourf(varx, vary, PS.maskOri[:,4].reshape(PS.nps), 1, hatches = [" ", "O"], alpha = 0) 
        #plt.yticks([-4,-3,-2,-1,0,1,2,3,4,5,6], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])

        if 'tau_w' in namex:
            #print(invsymlog10(np.amax(varx)))
            ax.plot([0,0], [np.amin(vary), np.amax(vary)], c = 'w', lw = .5) #look at this!!!
            numTicks = 5
            #ax.plot([0,0], [np.amin(vary), np.amax(vary)], c = 'w', lw = .5, ls = ':') #look at this!!!
            FwVal = invsymlog10(np.linspace(np.amin(varx), np.amax(varx), numTicks))
            FwTicks = []
            for t in range(numTicks): FwTicks.append(f'{FwVal[t]:1.1f}')
            plt.xticks(np.linspace(np.amin(varx), np.amax(varx), numTicks), FwTicks)
            #print(invsymlog10(np.linspace(np.amin(varx), np.amax(varx), 11)))
            #print(type(invsymlog10(np.linspace(np.amin(varx), np.amax(varx), 11))))
            #print(np.linspace(symlog10(np.amin(varx)), symlog10(np.amax(varx)), 11))
            #print(np.amax(varx))
            xl = r'$\tau_w$'
        if 'Q' in namex:
            plt.xticks([1,2,3,4], ['$10^1$', '$10^2$', '$10^3$', '$10^4$'])
            xl = r'$Q$'
        if 'H' in namex:
            plt.xticks([0,1,2], ['$1$', '$10^1$', '$10^2$'])
            xl = r'$H$'
        if 'K_M' in namex:
            plt.xticks([-4,-3,-2,-1,0], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
            xl = r'$K_M$'
        if 'K_H' in namex:
            plt.xticks([0,1,2,3,4,5], ['$1$','$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$'])
            xl = r'$K_H$'
        if 'Q' in namey:
            #print('hoi')
            yl = r'$Q$'
            plt.yticks([1,2,3,4], ['$10^1$', '$10^2$', '$10^3$', '$10^4$'])
        if 'H' in namey:
            yl = r'$H$'
            plt.yticks([0,1,2], ['$1$', '$10^1$', '$10^2$'])
        if 'K_M' in namey:
            yl = r'$K_M$'
            plt.yticks([-4,-3,-2,-1,0], ['$10^{-4}$','$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
        if 'K_H' in namey:
            yl = r'$K_H$'
            plt.yticks([0,1,2,3,4,5], ['$1$','$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$'])
        if a in [0,1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        if a in [1,3]:
            plt.setp(ax.get_yticklabels(), visible=False)
        
        if a in [2,3]:            
            ax.set_xlabel(xl)
        if a in [0,2]:
            ax.set_ylabel(yl)

def plotNDim3(PS):
    sRa, sFr, sFw = np.log10(PS.Ra.ravel()), np.log10(PS.Fr.ravel()), symlog10(PS.Fw.ravel())
    totMask = PS.totMask
    
    Reg =  np.array(PS.Reg.ravel())
    Reg[totMask] = 0
    #plt.plot(Reg)
    quan = [np.count_nonzero(np.round(Reg) == ind) for ind in range(5)]

    fig = plt.figure()
    fig.suptitle('3D Parameter Space (log axes)')
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    f1 = ax.scatter(sRa, sFr, sFw, c = PS.Reg.ravel(), cmap = plt.get_cmap('plasma'))
    cb = plt.colorbar(f1, ax = ax, ticks = [1 ,2, 3, 4])
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title('Regime: #' + str(quan))
    
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    f2 = ax.scatter(sRa, sFr, sFw, c = PS.Phi_0.ravel(), cmap = 'Blues')
    plt.colorbar(f2, ax = ax)
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title(r'Mouth stratification $\Phi_0$')
    #print(np.min(PS.Phi_0)) correct
    
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    f3 = ax.scatter(sRa, sFr, sFw, c = np.log10(PS.Ls.ravel()), cmap = 'Blues') #np.log10(PS.Ls.ravel())plt.get_cmap('plasma')) #np.log10(PS.Ls.ravel())
    plt.colorbar(f3, ax = ax)
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title(r'Salt intrusion length $L_s$ (log)')
    
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    f4 = ax.scatter(sRa, sFr, sFw, c = PS.domTermPlot, cmap = plt.get_cmap('plasma')) #np.log10(PS.Ls.ravel())
    plt.colorbar(f4, ax = ax)
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title('Dominant Terms: #' + str(PS.domTerm))

def plotDim3(PS, dimDict):
    names = dimDict['pars'] #tuple of variables that were varied.
    pars = []
    for i in range(3):
        if names[i] == 'tau_w':
            pars.append(symlog10(dimDict[names[i]]))
        else:
            pars.append(np.log10(dimDict[names[i]]))
            
    cmap = 'Blues'
    
    Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xs)), np.squeeze(PS.Reg)
    quan = [np.count_nonzero(np.round(Reg) == ind) for ind in range(5)]
    
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')    #fig.suptitle('Sensitivity: Wind and Depth')

    f1 = ax.scatter(*pars, c = PS.Reg.ravel(), cmap = plt.get_cmap('plasma'))
    cb = plt.colorbar(f1, ax = ax, ticks = [1 ,2, 3, 4])
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    ax.set_title('Regime: #' + str(quan))
    
    #fig = plt.figure()
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    f2 = ax.scatter(*pars, c = PS.Phi_0.ravel(), cmap = cmap)
    plt.colorbar(f2, ax = ax)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    ax.set_title(r'Mouth stratification $\Phi_0$')
    
    #print(np.min(PS.Phi_0)) correct
    #fig = plt.figure()
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    f3 = ax.scatter(*pars, c = np.log10(PS.Ls.ravel()), cmap = cmap) #np.log10(PS.Ls.ravel())
    plt.colorbar(f3, ax = ax)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    ax.set_title(r'Salt intrusion $L_s$ (log)')
    
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    f4 = ax.scatter(*pars, c = PS.domTermPlot, cmap = plt.get_cmap('plasma')) #np.log10(PS.Ls.ravel())
    plt.colorbar(f4, ax = ax)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    ax.set_title('Dominant Terms')
    
    
def plot3D(PS):
    sRa, sFr, sFw = np.log10(PS.Ra.ravel()), np.log10(PS.Fr.ravel()), symlog10(PS.Fw.ravel())
    totMask = PS.totMask
    
    Reg =  np.array(PS.Reg.ravel())
    Reg[totMask] = 0
    quan = [np.count_nonzero(np.round(Reg) == ind) for ind in range(5)]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    f1=ax.scatter(sRa, sFr, sFw, c = PS.Reg.ravel(), cmap = plt.get_cmap('plasma'))
    cb = plt.colorbar(f1, ax = ax, ticks = [1 ,2, 3, 4])
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title('Regime: #' + str(quan))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    f2 = ax.scatter(sRa, sFr, sFw, c = PS.Phi_0.ravel(), cmap = 'Blues')
    plt.colorbar(f2, ax = ax)
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title(r'Mouth stratification $\Phi_0$')
    
    #print(np.min(PS.Phi_0)) correct
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    f3 = ax.scatter(sRa, sFr, sFw, c = np.log10(PS.Ls.ravel()), cmap = 'Blues') #np.log10(PS.Ls.ravel())
    plt.colorbar(f3, ax = ax)
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title(r'Salt intrusion $L_s$ (log)')
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    f4 = ax.scatter(sRa, sFr, sFw, c = PS.Sb_X0.ravel(), cmap = 'Blues') #np.log10(PS.Ls.ravel())
    plt.colorbar(f4, ax = ax)
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title(r'Salinity gradient $\bar{\Sigma}_{X0}$')
    
    labT = ['G-G', 'G-R', 'G-W', 'R-R', 'R-W', 'W-W', 'DI', '|FL|']
    
    for ind in range(len(PS.aTT)):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        f5 = ax.scatter(sRa, sFr, sFw, c = PS.aTT[ind].ravel(), cmap = 'Blues') #np.log10(PS.Ls.ravel())
        plt.colorbar(f5, ax = ax)
        ax.set_xlabel('Ra')
        ax.set_ylabel('Fr')
        ax.set_zlabel('Fw')
        ax.set_title(labT[ind])
    
def plotSModel(pp,SM):
    X, Xp, sigmap, sigma = SM.X, SM.Xp, SM.sigmap, SM.sigma
    S, Sb = SM.S, SM.Sb
    
    Sb_X, r = SM.Sb_X, SM.r
    
    col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    
    U, W = SM.U, SM.W
    TX = SM.TX

    labT = ['G-G', 'G-R', 'G-W', 'R-R', 'R-W', 'W-W', 'DI', '|FL|']
    cmap = 'Blues'
    
    xnew = np.linspace(np.min(X), 0, np.max(np.shape(X)))
    Xp_interp, _ = np.meshgrid(xnew, sigma)
    
    U_interp  = griddata((np.ravel(sigmap), np.ravel(Xp)), np.ravel(U) , (sigmap, Xp_interp), method='linear')
    W_interp  = griddata((np.ravel(sigmap), np.ravel(Xp)), np.ravel(W) , (sigmap, Xp_interp), method='linear')

    
    _, axs = plt.subplots(3,2)
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
    axs[1,0].contour(f1, levels = np.linspace(0,1,10), colors = 'k', linewidths = 0.5)
    axs[1,0].contour(f1, levels = np.linspace(0,1,50), colors = 'k', linewidths = 0.5, l = 0.2)
    axs[1,0].title.set_text(f'Salinity: Negative = {SM.mask[3]} and Unstable = {SM.mask[4]}')
    axs[1,0].set_xlabel(r'$X$')
    axs[1,0].set_ylabel(r'$\sigma$')
    #if np.amin(S) < 0:
        #axs[1,0].contourf(Xp, sigmap, S, levels = [np.amin(S), 0, np.amax(S)], cmap = cmap, corner_mask = True, hatches=["//", ""], l = 0)
        #axs[1,0].contour(f1, levels = [0], colors='w', linewidths = 1.5)
    plt.colorbar(f1, ax=axs[1,0])
    
    mag = np.sqrt(U_interp[1:-1, 1:-1]**2 + W_interp[1:-1, 1:-1]**2)
    f2 = axs[2,0].contourf(Xp, sigmap, U, 50, cmap = cmap, corner_mask = True)
    axs[2,0].streamplot(Xp_interp[1:-1, 1:-1], sigmap[1:-1, 1:-1], U_interp[1:-1, 1:-1], W_interp[1:-1, 1:-1], density = 1, color='k', linewidth = mag/mag.max())
    axs[2,0].title.set_text('Flow')
    axs[2,0].set_xlabel(r'$X$')
    axs[2,0].set_ylabel(r'$\sigma$')
    axs[2,0].contour(f2, levels = [0], colors='w', linewidths = 1.5)
    plt.colorbar(f2, ax=axs[2,0])  
    
    for t in range(len(TX)):
        if t<7:
            axs[0,1].plot(X, TX[t], label = labT[t], color = col[t])
        else:
            axs[0,1].plot(X, np.abs(TX[t]), '-.', label = labT[t], color = col[t])
    axs[0,1].title.set_text('Salt transport')
    axs[0,1].set_xlabel(r'$X$')
    axs[0,1].set_ylabel('Relative contribution')
    axs[0,1].legend()
    axs[0,1].grid(True)
    

    # Change the bar colors here
    SM.T[-1] = np.abs(SM.T[-1])
    axs[1,1].bar(labT, SM.T, color=col)
    axs[1,1].bar(labT[-1], SM.T[-1], color=col[-1], hatch = '//')
    axs[1,1].title.set_text('Integrated salt transport: Regime = '+ np.array2string(SM.Reg))
    #axs[1,1].set_xlabel(r'$X$')
    axs[1,1].set_ylabel('Relative contribution')
    axs[1,1].grid(True)

    sbxmin = min([SM.Exx[1], np.min(Sb_X)])
    sbxmax = max([SM.Exx[0], np.max(Sb_X)])

    Sb_Xplot = np.linspace(sbxmin, sbxmax, 201)
    Sbplot = np.polyval([SM.a/SM.d,SM.b/SM.d,SM.c/SM.d,0], Sb_Xplot)
    
    axs[2,1].plot(Sb_Xplot, Sbplot, ls = 'dotted', label = 'Curve')
    axs[2,1].plot(Sb_X, Sb, lw = 2, label = 'Realised')
    axs[2,1].scatter(SM.Exx, SM.Exy, marker = 'o', label = 'J = 0 or H = 0')
    axs[2,1].title.set_text(r'$\bar{\Sigma}_X - \bar{\Sigma}$ Curve - ' + f'Non-unique = {SM.mask[2]}')
    axs[2,1].set_xlabel(r'$\bar{\Sigma}_X$')
    axs[2,1].set_ylabel(r'$\bar{\Sigma}$')
    axs[2,1].grid(True)
    axs[2,1].legend()

def plotModel3(pp,SM):
    X, Xp, sigmap, sigma = SM.X, SM.Xp, SM.sigmap, SM.sigma
    S, Sb = SM.S, SM.Sb
    
    Sb_X, r = SM.Sb_X, SM.r
    
    labT = ['GG', 'GR', 'GW', 'RR', 'RW', 'WW', 'D', '|F|']
    col = ['g', 'g', 'b', 'g', 'b', 'b', 'r', 'k']
    col = ['tab:green','tab:green', 'tab:orange', 'tab:green','tab:orange','tab:orange', 'tab:blue', 'tab:gray']
    ls = ['-', '--', '--', '-.', '-.', '-', '-', '-']
    U, W = SM.U, SM.W
    TX = SM.TX

    cmap = 'Blues'
    
    xnew = np.linspace(np.min(X), 0, np.max(np.shape(X)))
    Xp_interp, _ = np.meshgrid(xnew, sigma)
    
    U_interp  = griddata((np.ravel(sigmap), np.ravel(Xp)), np.ravel(U) , (sigmap, Xp_interp), method = 'linear')
    W_interp  = griddata((np.ravel(sigmap), np.ravel(Xp)), np.ravel(W) , (sigmap, Xp_interp), method = 'linear')

    

    fig = plt.figure(constrained_layout=True)
    s3 = fig.add_gridspec(ncols=1, nrows=3)
    for sp in [0,1,2]:
        ax = fig.add_subplot(s3[sp])
        #print(S)
        if sp == 0:
            S[-1,0] = 0
            a1 = ax.contourf(Xp, sigmap, S, 50, cmap=cmap, corner_mask = True)
            ax.contour(a1, levels = np.linspace(0,1,10), colors = 'k', linewidths = 0.5)
            ax.plot(X, Sb - 1, 'w', lw = 2, ls = '-.')
            ax.set_title(r'Salinity $\Sigma(X,\sigma)$ and $\bar{\Sigma}(X)$')
            cb=plt.colorbar(a1, ax=ax)
            cb.set_ticks([0.0, .2, .4, .6, .8, 1.0])
        if sp == 1:
            lev =  np.linspace(np.amin(U) ,np.amax(U),10)
            a2 = ax.contourf(Xp, sigmap, U, 50, cmap = 'RdBu', corner_mask = True, vmin = min(np.amin(U), -np.amax(U)), vmax = max(np.amax(U), -np.amin(U)))
            ax.contour(a2, levels = lev, colors = 'k', linewidths = 0.5)
            ax.set_title(r'Flow $U(X,\sigma)$')
            cb=plt.colorbar(a2, ax=ax)
            tic = np.linspace(np.amin(U), np.amax(U), 6)
            print(tic)
            cb.set_ticks(tic)
            cb.set_ticklabels(["%.3f" % ti for ti in tic])
        if sp == 2:
            ax.plot(X, np.abs(TX[7]), label = labT[7], color = col[7], ls = ls[7])
            for t in range(len(TX)-1):
                ax.plot(X, TX[t], label = labT[t], color = col[t], ls = ls[t])
                    
            ax.title.set_text(r'Scaled salt transports $T_i(X)$')
            ax.set_xlabel(r'$X$')
            ax.set_ylabel('Rel. magnitude')
            ax.grid(True)
        ax.set_ylabel(r'$\sigma$')
        if sp in[0,1]:
            ax.set_yticks([0, -.25, -.5, -.75, -1])
        else:
            #ax.set_yticks([0, .25, .5, .75, 1])
            pass

    
    '''
    # Change the bar colors here
    SM.T[-1] = np.abs(SM.T[-1])
    axs[1,1].bar(labT, SM.T, color=col)
    axs[1,1].bar(labT[-1], SM.T[-1], color=col[-1], hatch = '//')
    axs[1,1].title.set_text('Integrated salt transport: Regime = '+ np.array2string(SM.Reg))
    #axs[1,1].set_xlabel(r'$X$')
    axs[1,1].set_ylabel('Relative contribution')
    axs[1,1].grid(True)

    sbxmin = min([SM.Exx[1], np.min(Sb_X)])
    sbxmax = max([SM.Exx[0], np.max(Sb_X)])

    Sb_Xplot = np.linspace(sbxmin, sbxmax, 201)
    Sbplot = np.polyval([SM.a/SM.d,SM.b/SM.d,SM.c/SM.d,0], Sb_Xplot)
    
    axs[2,1].plot(Sb_Xplot, Sbplot, ls = 'dotted', label = 'Curve')
    axs[2,1].plot(Sb_X, Sb, lw = 2, label = 'Realised')
    axs[2,1].scatter(SM.Exx, SM.Exy, marker = 'o', label = 'J = 0 or H = 0')
    axs[2,1].title.set_text(r'$\bar{\Sigma}_X - \bar{\Sigma}$ Curve - ' + f'Non-unique = {SM.mask[2]}')
    axs[2,1].set_xlabel(r'$\bar{\Sigma}_X$')
    axs[2,1].set_ylabel(r'$\bar{\Sigma}$')
    axs[2,1].grid(True)
    axs[2,1].legend()
    '''
    
def plotModel(SM, PS):
    X, Xp, sigmap, sigma = SM.X, SM.Xp, SM.sigmap, SM.sigma
    S, Sb = SM.S, SM.Sb
    
    Sb_X, r = SM.Sb_X, SM.r
    
    col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    
    U, W = SM.U, SM.W
    T = SM.T

    labT = ['G-G', 'G-R', 'G-W', 'R-R', 'R-W', 'W-W', 'DI', '|FL|']
    cmap = 'Blues'
    
    xnew = np.linspace(np.min(X), 0, np.max(np.shape(X)))
    Xp_interp, _ = np.meshgrid(xnew, sigma)
    
    U_interp  = griddata((np.ravel(sigmap), np.ravel(Xp)), np.ravel(U) , (sigmap, Xp_interp), method='linear')
    W_interp  = griddata((np.ravel(sigmap), np.ravel(Xp)), np.ravel(W) , (sigmap, Xp_interp), method='linear')

    
    _, axs = plt.subplots(3,3)
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
    axs[1,0].contour(f1, levels = np.linspace(0,1,10), colors = 'k', linewidths = 0.5)
    axs[1,0].contour(f1, levels = np.linspace(0,1,50), colors = 'k', linewidths = 0.5, l = 0.2)
    axs[1,0].title.set_text(f'Salinity: Negative = {SM.maskNEG} and Unstable = {SM.maskIS}')
    axs[1,0].set_xlabel(r'$X$')
    axs[1,0].set_ylabel(r'$\sigma$')
    if np.amin(S) < 0:
        axs[1,0].contourf(Xp, sigmap, S, levels = [np.amin(S), 0, np.amax(S)], cmap = cmap, corner_mask = True, hatches=["//", ""], l = 0)
        axs[1,0].contour(f1, levels = [0], colors='w', linewidths = 1.5)
    plt.colorbar(f1, ax=axs[1,0])
    
    mag = np.sqrt(U_interp[1:-1, 1:-1]**2 + W_interp[1:-1, 1:-1]**2)
    f2 = axs[2,0].contourf(Xp, sigmap, U, 50, cmap = cmap, corner_mask = True)
    axs[2,0].streamplot(Xp_interp[1:-1, 1:-1], sigmap[1:-1, 1:-1], U_interp[1:-1, 1:-1], W_interp[1:-1, 1:-1], density = 1, color='k', linewidth = mag/mag.max())
    axs[2,0].title.set_text('Flow')
    axs[2,0].set_xlabel(r'$X$')
    axs[2,0].set_ylabel(r'$\sigma$')
    axs[2,0].contour(f2, levels = [0], colors='w', linewidths = 1.5)
    plt.colorbar(f2, ax=axs[2,0])  
    
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
    

    # Change the bar colors here
    SM.aTT[-1] = -SM.aTT[-1]
    axs[1,1].bar(labT, SM.aTT, color=col)
    axs[1,1].bar(labT[-1], SM.aTT[-1], color=col[-1], hatch = '//')
    axs[1,1].title.set_text('Integrated salt transport')
    #axs[1,1].set_xlabel(r'$X$')
    axs[1,1].set_ylabel('Relative contribution')
    axs[1,1].grid(True)

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
    
    # From here, we continue with PS
    
    Ra, Fr, Fw, Phi_0, Ls, Reg = np.squeeze(PS.Ra), np.squeeze(PS.Fr), np.squeeze(PS.Fw), np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xs)), np.squeeze(PS.Reg)
    
    cfig = axs[0,2].contourf(Ra, Fr, Reg, 20, cmap = plt.get_cmap('plasma'))
    #axs[0,2].contour(cfig,  colors = 'k', linewidths = 0.5)
    #axs[0,2].contour(cfig, levels = [0,1], colors='k', linewidths = 1)
    #axs[0,2].contourf(Ra, Fr, Reg, levels = hl, vmin = minz, vmax = maxz, hatches = [".", "", "//"], l = 0)
    axs[0,2].scatter(SM.Ra, SM.Fr, color = 'k', marker = 'o')    #axs[1,2].clabel(cfigc, cfigc.levels, inline = True, fontsize = 10)       
    axs[0,2].title.set_text(r'Regime = '+ f'{float(SM.Reg):.1f}' + r' with $F_W$ = ' + f'{SM.Fw:.3f}')
    axs[0,2].set_xlabel(r'Ra')
    axs[0,2].set_ylabel(r'Fr')
    axs[0,2].set_xscale('log')
    axs[0,2].set_yscale('log')
    cb = plt.colorbar(cfig, ax = axs[0,2], ticks = [1 ,2, 3, 4])
    #cb.ax.set_yticklabels(['1', '2', '3', '4'])
    #cb = plt.colorbar(cfig, ax = axs[0,2])
    #cb.ax.set_yticklabels(['1', '2', '3', '4'])    
    
    
    hl = [np.min([np.amin(Phi_0),0]), np.max([np.amin(Phi_0),0]), np.min([np.amax(Phi_0),1]), np.max([np.amax(Phi_0),1])]
    hl.sort()
    minz, maxz = np.min(Phi_0), np.max(Phi_0)
    cfig = axs[1,2].contourf(Ra, Fr, Phi_0, 50, cmap = cmap, corner_mask = True, vmin = minz, vmax = maxz)
    axs[1,2].contour(cfig,  colors = 'k', linewidths = 0.5)
    axs[1,2].contour(cfig, levels = [0,1], colors='k', linewidths = 1)
    axs[1,2].contourf(Ra, Fr, Phi_0, levels = hl, vmin = minz, vmax = maxz, hatches = [".", "", "//"], l = 0)
    axs[1,2].scatter(SM.Ra, SM.Fr, color = 'k', marker = 'o')    #axs[1,2].clabel(cfigc, cfigc.levels, inline = True, fontsize = 10)       
    axs[1,2].title.set_text(r'Stratification $\Phi_0$')
    axs[1,2].set_xlabel(r'Ra')
    axs[1,2].set_ylabel(r'Fr')
    axs[1,2].set_xscale('log')
    axs[1,2].set_yscale('log')
    plt.colorbar(cfig, ax = axs[1,2])

    minz, maxz = np.amin(Ls), np.amax(Ls)
    #print(np.amin(Ls))
    #cfig2 = axs[2,2].contourf(Ra, Fr, Ls, 20, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True, vmin = minz, vmax = maxz)
    cfig2 = axs[2,2].contourf(Ra, Fr, Ls, 50, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True)
    axs[2,2].contour(cfig2,  colors = 'k', linewidths = 0.5)
    axs[2,2].contour(cfig2, levels = [1e-3, 1e-2, 1e-1, 1, 10, 100], colors='k', linewidths = 1)
    #axs[1,2].contourf(Ra, Fr, Xs, levels = [np.min([np.amin(Xs),-1]), np.max([np.amin(self.Xs),-1]),0], vmin = minz, vmax = maxz, hatches = [".", "", "//"], l = 0)
    axs[2,2].scatter(SM.Ra, SM.Fr, color = 'k', marker = 'o')    #axs[1,2].clabel(cfigc, cfigc.levels, inline = True, fontsize = 10)       
    axs[2,2].title.set_text(r'Salt Intrusion $L_s$')
    axs[2,2].set_xlabel(r'Ra')
    axs[2,2].set_ylabel(r'Fr')
    axs[2,2].set_xscale('log')
    axs[2,2].set_yscale('log')
    plt.colorbar(cfig2, format = ticker.LogFormatterMathtext(), ax = axs[2,2])
    
    
def plotCubics4(pp,SMT):
    fig = plt.figure(constrained_layout=True)
    s = fig.add_gridspec(ncols=2, nrows=2)
    
    for i in range(4):
        SM = SMT[i]
        ax = fig.add_subplot(s[i])
        sbxmin = min([SM.Exx[0], SM.Exx[1], SM.Exx[2], np.min(SM.Sb_X)])
        sbxmax = max([SM.Exx[0], SM.Exx[1], SM.Exx[2], np.max(SM.Sb_X)])

        Sb_Xplot = np.linspace(sbxmin, sbxmax, 201)
        Sbplot = np.polyval([SM.a/SM.d,SM.b/SM.d,SM.c/SM.d,0], Sb_Xplot)
    
        ax.plot(Sb_Xplot, Sbplot, c = 'k', ls = 'dotted', lw = .5, label = 'Curve')
        ax.scatter(SM.Exx, SM.Exy, c = 'k', marker = 'o', label = 'J = 0 or H = 0')
        ax.plot(SMT[i].Sb_X, SMT[i].Sb, lw = 1.5, label = 'Realised')
    #axs[2,1].title.set_text(r'$\bar{\Sigma}_X - \bar{\Sigma}$ Curve - ' + f'Non-unique = {SM.mask[2]}')
        if i in [2,3]:
            ax.set_xlabel(r'$\bar{\Sigma}_X$')
        if i in [0,2]:
            ax.set_ylabel(r'$\bar{\Sigma}$')
        ax.grid(True)
    #axs[2,1].legend()
    
def plotCubics(pp,SMT):
    fig = plt.figure(constrained_layout=True)
    #s = fig.add_gridspec(ncols=2, nrows=2)
    
    #SM = SMT[i]
    #ax = fig.add_subplot(s[i])
    for SM in SMT:
        try: 
            sbxmin = min([SM.Exx[0], SM.Exx[1], SM.Exx[2], np.min(SM.Sb_X)])
            sbxmax = max([SM.Exx[0], SM.Exx[1], SM.Exx[2], np.max(SM.Sb_X)])

            Sb_Xplot = np.linspace(sbxmin, sbxmax, 201)
            Sbplot = np.polyval([SM.a/SM.d,SM.b/SM.d,SM.c/SM.d,0], Sb_Xplot)
    
            plt.plot(Sb_Xplot, Sbplot, c = 'k', ls = 'dotted', lw = .5)
            plt.scatter(SM.Exx, SM.Exy, c = 'k', marker = 'o')
            plt.plot(SM.Sb_X, SM.Sb, lw = 1.5, label = f'$Fw$ = {SM.Fw:.1f}')
        except:
            pass
    plt.xlabel(r'$\bar{\Sigma}_X$')
    plt.ylabel(r'$\bar{\Sigma}$')
    plt.title(r'Cubic curves, varying $Fw$')
    plt.xlim([0, 1e-4])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    
def plotNDimEst(pp,PST):
    fig = plt.figure(constrained_layout=True)
    s = fig.add_gridspec(ncols=2, nrows = len(PST))
    i = 0
    for PS in PST:
        ndd = PS.nondimDict
        namex = ndd['pars'][0]
        if PS.gp['Ori']:
            namex = namex+'Ori'

        if 'Fw' in namex:
            varx = symlog10(getattr(PS, namex).reshape(PS.nps))
        else:
            varx = np.log10(getattr(PS, namex).reshape(PS.nps))
        
        Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xs)), np.squeeze(PS.Reg)
        LsT = PS.LsT

        Regcolor = np.array([Reg, Reg])
        extentp = (np.amin(varx), np.amax(varx), np.amin(Phi_0), np.amax(Phi_0))
        Ll, Lu = 1/2*np.amin(Ls), 2*np.amax(Ls)
        extentl = (np.amin(varx), np.amax(varx), Ll, Lu)
        #x, y = x.ravel(), y.ravel()
        
        LD = np.ma.masked_outside(LsT[:,6], Ll, Lu)
        LGG = np.ma.masked_outside(LsT[:,0], Ll, Lu)
        LWW = np.ma.masked_outside(LsT[:,5], Ll, Lu)
        LGW = np.ma.masked_outside(LsT[:,-1], Ll, Lu)
        
        ax = fig.add_subplot(s[i,0])
        ax.imshow(Regcolor, extent = extentp, origin = 'upper', aspect = 'auto')
        ax.plot(varx, Phi_0, lw = 2, c = 'k')
        if i == 0:
            ax.title.set_text(r'Stratification $\Phi_0$')
        if 'Fw' in namex:
            ax.plot([0,0], np.array([np.amin(Phi_0), np.amax(Phi_0)]), c = 'w', lw=.5)
        nt = 7
        FwVal = invsymlog10(np.linspace(np.amin(varx), np.amax(varx), nt))
        FwTicks = []
        for t in range(nt): FwTicks.append(f'{FwVal[t]:.2n}')
        #ax.set_xticks(np.linspace(np.amin(varx), np.amax(varx), 11), FwTicks)
        #axs.set_xticks(np.linspace(np.amin(varx), np.amax(varx), 11), FwTicks)
        #plt.xticks(np.linspace(np.amin(varx), np.amax(varx), 11), FwTicks)
        plt.xticks(np.linspace(np.amin(varx), np.amax(varx), nt), FwTicks)
        ax.set_title(f'{PS.name} : $\Phi_0$')
        axs = fig.add_subplot(s[i,1])
        axs.plot(varx, Ls, lw = 4, c = 'k', label = r'$L_s$')
        if i == 0:
            axs.title.set_text(r'Salt intrusion $\Lambda_s$')
        axs.imshow(Regcolor, extent = extentl, origin = 'upper', aspect = 'auto')
        axs.plot(varx, LD, ls = '-', lw = 1, c = 'w', label = 'Dispersive') #Dispersive Regime
        axs.plot(varx, LGG, ls = '--', lw = 1, c = 'w', label = 'Chatwin') #Chatwin Regime
        axs.plot(varx, LWW, ls = '-.', lw = 1, c = 'w', label = 'Wind-driven') #WW Regime
        axs.plot(varx, LGW, ls = ':', lw = 1, c = 'w', label = 'GW-GG') #WW Regime
        axs.set_title(f'{PS.name} : $\Lambda_s$')
        if 'Fw' in namex:
            axs.plot([0,0], np.array([Ll, Lu]), c = 'w', lw=.5)
        axs.set_yscale('log')
        #plt.xticks(np.linspace(np.amin(varx), np.amax(varx), nt), FwTicks)

        #ax.legend()
        
        if i<len(PST)-1:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(axs.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(ndd['pars'][0])
            axs.set_xlabel(ndd['pars'][0])

        plt.xticks(np.linspace(np.amin(varx), np.amax(varx), nt), FwTicks)
        i=i+1
