import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from matplotlib import ticker, cm
import matplotlib.tri as mtri
import matplotlib.colors as co


def globalParameters(**kwargs):
    gp = dict(BC = 1, #1 = partial Slip (dijkstra), 0 = zero friction, -1 = infinite friction (MacCready)
        alpha = 1/30, #Salinity fraction at landward end
        n = [1, 401, 25, 11], #0 parameters to vary, 1 parameter, 2 parameters, 3 parameters.
        SMGrid = [1001,21], #Single model run: grid.
        PSGrid = [51,11],
        tolNEG = 1,
        tolUN = 10,
        tolPrit = 10,
        Sc = 2.2
        )
    gp['C'] = addC(gp['BC'])
    for key, value in kwargs.items(): #Change the constant parameters
            gp[key] = value
    return gp

def solveCubic(r, opt = 'minusOne'):
    '''Computes single real solution based on parameter values'''
    solC = np.roots(r)
    D3 = computeD3(r[0],r[1],r[2],r[3])
    maskNU = (D3>=0)
    if ~maskNU: #one real solution
        sol = float(np.real(solC[np.isreal(solC)]))
    if maskNU: #two or three real solutions
        #maskNU0 = True
        if opt == 'minusOne':
            sol = -1
        elif opt == 'max':
            sol = np.amax(solC)
        elif opt == 'min':
            sol = np.amin(solC)
        elif opt == 'middle':
            solS = np.sort(solC)
            sol = solS[1]
    return sol, maskNU

def computeD3(a,b,c,d):
    D3 = b**2*c**2 - 4*a*c**3 - 4*b**3*d - 27*a**2*d**2 + 18*a*b*c*d
    return D3

def computeD2(a,b,c):
    D2 = b**2 - 4*a*c
    return D2

def computeLocalExtrema(a,b,c,d):
    '''Returns the discriminant and local extrema of the S - S_X curve'''
    D2 = computeD2(3*a, 2*b, c)
    bpx = -b/(3*a) #x where second derivative is zero
    bpy = a*bpx**3 + b*bpx**2 + c*bpx + d #y where second derivative is zero
    xright, xleft, yright, yleft = bpx, bpx, bpy, bpy #no local extrema
    if D2 >= 0:
        xright = (-2*b + np.sqrt(D2))/(6*a)#x where first derivative is zero
        xleft = (-2*b - np.sqrt(D2))/(6*a)#x where first derivative is zero
        yright = a*xright**3 + b*xright**2 + c*xright + d#y where first derivative is zero
        yleft = a*xleft**3 + b*xleft**2 + c*xleft + d#y where first derivative is zero
    #xright, xleft, yright, yleft = bpx, bpx, bpy, bpy
    return D2, [xright,xleft,bpx], [yright,yleft,bpy]

def computeLocalExtremaVect(a,b,c,d):
    '''Returns the discriminant and local extrema of the S - S_X curve'''
    D2 = computeD2(3*a, 2*b, c)
    bpx = -b/(3*a) #x where second derivative is zero
    bpy = a*bpx**3 + b*bpx**2 + c*bpx + d #y where second derivative is zero
    xright, xleft, yright, yleft = bpx, bpx, bpy, bpy #no local extrema
    xright[D2>=0] = (-2*b[D2>=0] + np.sqrt(D2[D2>=0]))/(6*a)#x where first derivative is zero
    xleft[D2>=0] = (-2*b[D2>=0] - np.sqrt(D2[D2>=0]))/(6*a)#x where first derivative is zero
    yright[D2>=0] = a*xright[D2>=0]**3 + b[D2>=0]*xright[D2>=0]**2 + c[D2>=0]*xright[D2>=0] + d#y where first derivative is zero
    yleft[D2>=0] = a*xleft[D2>=0]**3 + b[D2>=0]*xleft[D2>=0]**2 + c[D2>=0]*xleft[D2>=0] + d#y where first derivative is zero
    #xright, xleft, yright, yleft = bpx, bpx, bpy, bpy
    return D2, [xright,xleft,bpx], [yright,yleft,bpy]

def dim2nondim(dimDict):
    Q, H, K_M, tau_w, K_H = dimDict['Q'], dimDict['H'], dimDict['K_M'], dimDict['tau_w'], dimDict['K_H']
    s_0, g, beta, rho_0 ,B= dimDict['s_0'], dimDict['g'], dimDict['beta'], dimDict['rho_0'], dimDict['B']
    
    c = np.sqrt(g*beta*s_0*H)
    ur = Q/(B*H)
    uw = tau_w*H/(rho_0*K_M) 
    
    Fr = ur/c
    Fw = uw/c
    Ra = c**2*H**2/(K_M*K_H)
    
    nondimDict = dict(Ra = Ra, Fr = Fr, Fw = Fw, Sc = dimDict['Sc'])
    return nondimDict

def makeDicts(gp, *args, **kwargs):
    #First default values by Dijkstra and Schuttelaars (2021)
    dd = dict(Q = 1000, 
        H = 20,  
        K_M = 1e-3, 
        tau_w = 0,
        K_H = 250, 
        s_0 = 30.0, g = 9.81, beta = 7.6e-4, rho_0 = 1000.0, B = 1000.0, Sc = gp['Sc']
        )
    
    for key, value in kwargs.items(): #Change the constant parameters
        if key not in ['Ralston', 'Sverdrup']:
            dd[key] = value
        
    if len(args) == 0:
        ndd = dim2nondim(dd)
        ndd['pars'] = 'None'
        ndd['nps'] = 1
        return dd, ndd
    elif len(args) == 1:
        n = gp['n'][1]
    elif len(args) == 2:
        n = gp['n'][2]
    elif len(args) == 3:
        n = gp['n'][3]
    else:
        print('Wrong function input')
        return        
        #ndd['nps'] = n
        
    #Default sweep ranges: Independent variables.
    Q = np.logspace(0,3,n)
    H = np.logspace(0,3,n)
    tau_w = np.concatenate((-np.logspace(0,-5, int((n-1)/2)), np.array([0]), np.logspace(-5,0,int((n-1)/2))))
    
    # Make mixing dependent on H or tau_w, not the other way around!!
    if ('Ralston' not in kwargs) and ('Sverdrup' not in kwargs):
        K_M = np.logspace(-6,0,n)
        K_H = np.logspace(0,5,n)
    elif 'Ralston' in kwargs:
        K_M, K_H = Ralston(kwargs['Ralston'], H, dd['B'])
        if 'Sverdrup' in kwargs:
            K_M = Sverdrup(tau_w, K_M, kwargs['Sverdrup'])
    elif 'Sverdrup' in kwargs:
        K_M = Sverdrup(tau_w, dd['K_M'], kwargs['Sverdrup'])
        K_H = dd['K_H']
    
    if len(args) == 1:
        varx = eval(args[0])
        dd[args[0]] = varx
        if ('Ralston' in kwargs) or ('Sverdrup' in kwargs):
            dd['K_M'], dd['K_H'] = K_M, K_H
        ndd = dim2nondim(dd)
        ndd['nps'] = n
        
    if len(args) == 2:
        varx, vary = np.meshgrid(eval(args[0]), eval(args[1])) #bad practice but in this case not very harmful
        varx, vary = varx.flatten(), vary.flatten()
        dd[args[0]] = varx
        dd[args[1]] = vary
        if 'Ralston' in kwargs:
            if args[0] == 'H':
                dd['K_M'], dd['K_H'] = Ralston(kwargs['Ralston'], varx, dd['B'])
            if args[1] == 'H':
                dd['K_M'], dd['K_H'] = Ralston(kwargs['Ralston'], vary, dd['B'])
                # Sverdrup!!!!!
            #dd['K_M'], dd['K_H'] = K_M, K_H
        ndd = dim2nondim(dd)
        ndd['nps'] = [n, n]

    if len(args) == 3:
        varx, vary, varz = np.meshgrid(eval(args[0]), eval(args[1]), eval(args[2])) #bad practice but in this case not very harmful
        varx, vary, varz = varx.flatten(), vary.flatten(), varz.flatten()
        dd[args[0]] = varx
        dd[args[1]] = vary
        dd[args[2]] = varz
        ndd = dim2nondim(dd)
        ndd['nps'] = [n, n, n]

    #ndd = dim2nondim(dd)
    dd['pars'] = args
    ndd['pars'] = args
    return dd, ndd

def makeNDDict(gp, *args, **kwargs):
    dd = dict(Q = 1000, 
        H = 20,  
        K_M = 1e-3, 
        tau_w = 0,
        K_H = 250, 
        s_0 = 30.0, g = 9.81, beta = 7.6e-4, rho_0 = 1000.0, B = 1000.0, Sc = 2.2
        )
    ndd = dim2nondim(dd) #Default, also for nonDim model run.
    
    for key, value in kwargs.items(): #Change the constant parameters
        if key not in ['Ralston', 'Sverdrup']:
            ndd[key] = value
        
    if len(args) == 0:
        ndd = dim2nondim(dd)
        ndd['pars'] = 'None'
        ndd['nps'] = 1
        return ndd
    elif len(args) == 1:
        n = gp['n'][1]
    elif len(args) == 2:
        n = gp['n'][2]
    elif len(args) == 3:
        n = gp['n'][3]
    else:
        print('Wrong function input')
        return        
        #ndd['nps'] = n
        
    #Default sweep ranges: Independent variables.
    Ra = np.logspace(0, 6, n)    
    Fr = np.logspace(np.log10(5) - 5 , np.log10(1) , n)
    if n > 1:
        if n % 2 == 0: n = n + 1
        Fw = np.concatenate((-np.logspace(1.5,-4, int((n-1)/2)), np.array([0]), np.logspace(-4,1.5,int((n-1)/2))))

    if len(args) == 1:
        varx = eval(args[0])
        ndd[args[0]] = varx
        ndd['nps'] = n
        
    if len(args) == 2:
        varx, vary = np.meshgrid(eval(args[0]), eval(args[1]), indexing = 'ij') #bad practice but in this case not very harmful
        varx, vary = varx.flatten(), vary.flatten()
        ndd[args[0]] = varx
        ndd[args[1]] = vary
        ndd['nps'] = [n, n]

    if len(args) == 3:
        varx, vary, varz = np.meshgrid(eval(args[0]), eval(args[1]), eval(args[2]), indexing = 'ij') #bad practice but in this case not very harmful
        varx, vary, varz = varx.flatten(), vary.flatten(), varz.flatten()
        ndd[args[0]] = varx
        ndd[args[1]] = vary
        ndd[args[2]] = varz
        ndd['nps'] = [n, n, n]

    #ndd = dim2nondim(dd)
    ndd['pars'] = args
    ndd['Sc'] = gp['Sc']

    #Ra, Fr, Fw = np.meshgrid(Ra1, Fr1, Fw1, indexing='ij')
    #nondimDict = dict(Ra = Ra.flatten(),Fr = Fr.flatten(), Fw = Fw.flatten(), Sc = dd['Sc'], nps = nps)
    return ndd

def Ralston(u_T, H, B):
    K_M = 0.028*0.0026*u_T*H
    K_H = 0.035*u_T*B
    return K_M, K_H

def Sverdrup(tau_w, K_M0, beta):
    K_M = K_M0 + beta*np.abs(tau_w)
    return K_M

def computeMask(b,c, Ra, Fr, Fw, Sc, Sb_X0, ra, gp):
    '''Compute relevant masks (after non-unique Sb_X0 and Xa have already been masked). Notice that the Dijkstra slip condition is used.'''
    C = gp['C']
    r = np.linspace(ra, 0, gp['PSGrid'][0])
    X = 3/2*Sb_X0**2*(np.exp(-2*r)-1) + 2*b*Sb_X0*(np.exp(-r)-1) - c*r
    Sb = Sb_X0**3*np.exp(-3*r) + b*Sb_X0**2*np.exp(-2*r) + c*Sb_X0*np.exp(-r)
    Sb_X = Sb_X0*np.exp(-r)
    Sb_XX = Sb_X/(3*Sb_X**2 + 2*b)
    P40 = -7/300
    P50 = -23/7200
    P60 = -11/600
    P4 = -7/300 + 1/10 - 1/20
    P5 = -23/7200 + 1/60 - 3/160 - 1/120*-1
    P6 = -11/600 + 3/20 -1/6 + 1/20     
    #D2, x, y = computeLocalExtrema(1,b,c,0) # D2, [xright,xleft,bpx], [yright,yleft,bpy]
    SXsurface = Sb + C[12]*Sc**(2/3)*Fr**(1/3)*(Fr*P40 + Fw*P60)*Sb_X + C[13]*Sc**(1/3)*Fr**(2/3)*P50*Sb_X**2
    SXbottom = Sb + C[12]*Sc**(2/3)*Fr**(1/3)*(Fr*P4 + Fw*P6)*Sb_X + C[13]*Sc**(1/3)*Fr**(2/3)*P5*Sb_X**2
    maskNonUnique = np.any(Sb[1:] <= Sb[:-1]) # 1 if there are local extrema (if there is non-monotonicity)
    maskUnStable = np.any((SXbottom - SXsurface < -gp['tolUN'])) # 1 if there exists a point of unstable stratification
    maskNegative = np.any((SXsurface < -gp['tolNEG'])) # 1 if there is negative salt
    
    nsigma = gp['PSGrid'][1]
    sigma = np.linspace(-1, 0, nsigma)
    _, sigmap = np.meshgrid(X,sigma)

    _, _, _, P4, P5, P6, _ = formFunctions(sigmap, gp['BC'])
    #Ubar = Fr*np.ones_like(sigmap)    
    #UR = Fr*P1
    #UG = C[12]*Sc**(-1/3)*Fr**(1/3)*P2*np.matlib.repmat(np.transpose(Sb_X),nsigma,1)
    #UW = Fw*P3   
    #temp = np.transpose(Sb)# np.matlib.repmat(np.transpose(Sb),nsigma,1) -> Start here!!!!
    #print(temp)
    #self.Sbar = np.matlib.repmat(np.transpose(Sb),nsigma,1) 
    SaccX = C[12]*Sc**(2/3)*Fr**(1/3)*(Fr*P4 + Fw*P6)*np.matlib.repmat(np.transpose(Sb_XX),nsigma,1) + C[13]*Sc**(1/3)*Fr**(2/3)*P5*np.matlib.repmat(np.transpose(2*Sb_X*Sb_XX),nsigma,1)
    #epsilon = .5
    #Choice 1: Integral rule
    #trapacc = 1/(nsigma-1)*(np.sum(np.abs(SaccX[1:-1,:]),0) + (np.abs(SaccX[0,:]) + np.abs(SaccX[-1,:]))/2) #trapezoidal rule
    #maskPritchard = np.any((trapacc > epsilon*Sb_X))
    #Choice 2: Max rule
    maxacc = np.amax(np.abs(SaccX),0)
    maskPritchard = np.any((maxacc > gp['tolPrit']*Sb_X))
    #print(np.shape(np.abs(SaccX[:,1:-2])))
    
    #U =  Ubar + UR + UG + UW
    #UX = C[12]*Sc**(-1/3)*Fr**(1/3)*P2*np.matlib.repmat(np.transpose(Sb_XX),nsigma,1)
    #self.Ubar, self.UR, self.UG, self.UW = Ubar, UR, UG, UW
    
    return bool(maskNonUnique), bool(maskUnStable), bool(maskNegative), bool(maskPritchard)

#def computeScaling()

def I0(x,a,b,c):
    ''' Compute the integral of 1/(au^2 + bu + c) and fill in x'''
    D = b**2 - 4*a*c
    
    if D < 0:
        sD = np.sqrt(-D)
        I = 2/sD*np.arctan((2*a*x + b)/sD)
    elif D > 0:
        sD = np.sqrt(D)
        I = 1/sD*np.log(np.abs((2*a*x + b - sD)/(2*a*x + b + sD)))
    elif D==0:
        I = -2/(2*a*x + b)
    return I

#I0 = np.vectorize(I0a, otypes=[np.ndarray])

def I1(x,a,b,c):
    ''' Compute the integral of u/(au^2 + bu + c) and fill in x'''
    I = 1/(2*a)*np.log(np.abs(a*x**2 + b*x + c)) - b/(2*a)*I0(x,a,b,c)
    return I

def I2(x,a,b,c):
    ''' Compute the integral of u^2/(au^2 + bu + c) and fill in x'''
    I = x/a - b/(2*a**2)*np.log(np.abs(a*x**2 + b*x + c)) + (b**2 - 2*a*c)/(2*a**2)*I0(x,a,b,c)
    return I

def I3(x,a,b,c):
    ''' Compute the integral of u^3/(au^2 + bu + c) and fill in x'''
    I = x**2/(2*a) - b*x/a**2 + (b**2-a*c)/(2*a**3)*np.log(np.abs(a*x**2 + b*x + c)) - b*(b**2 - 3*a*c)/(2*a**3)*I0(x,a,b,c)
    return I

def scoreIJ(TT):
    s1 = TT[0]/TT[6]
    s1 = 0.1 if s1 < 0.1 else s1
    s1 = 10 if s1 > 10 else s1
    if TT[2] > 0:
        s2 = (TT[4] + TT[2] + TT[5])/(TT[0]+TT[6])
        s2 = 0.1 if s2 < 0.1 else s2
        s2 = 10 if s2 > 10 else s2
    else:
        s2 = 0
    if TT[2] < 0: #up-estuary wind: import of salt
        s3 = (np.abs(TT[2])+np.abs(TT[4])+ np.abs(TT[5]))/(TT[0]+TT[6])
        s3 = 0.1 if s3 < 0.1 else s3
        s3 = 10 if s3 > 10 else s3
    else: #
        s3 = 0
    maxT = np.argmax(np.abs(TT[0:7]))
    return np.array([s1,s2,s3,maxT])

def computeRegime(s):
    if s[2]>.1:
        r = 3 + np.log10(s[2])/2 + 1/2
    elif s[1]>.1:
        r = 2 + np.log10(s[1])/2 + 1/2
    elif s[0]>.1:
        r = 1 + np.log10(s[0])/2 + 1/2
    else:
        r = np.array([1.0])
    return np.array([r])
    
def symlog10(x,l = 1/10000/np.log(10)):
    return np.sign(x)*np.log10(1.0 + np.abs(x/l))
    
def invsymlog10(y,l = 1/10000/np.log(10)):
    return np.sign(y)*l*(-1.0 + 10**np.abs(y))


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
        
def plotNDim(PS):
    ndd = PS.nondimDict
    if len(ndd['pars']) == 1:
        plotNDim1(PS)
    elif len(ndd['pars']) == 2:
        plotNDim2(PS)
    elif len(ndd['pars']) == 3:
        plotNDim3(PS)

def plotNDim1(PS):
    col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    labT = ['G-G', 'G-R', 'G-W', 'R-R', 'R-W', 'W-W', 'DI', '|FL|']
    ndd = PS.nondimDict
    namex = ndd['pars'][0]
    tm = np.transpose(np.array([PS.totMask, PS.totMask], dtype = bool))
    # Prepare background color for plots
    
    #Reg = ma.masked_array(Reg, mask = tm)
    
    LSReg = PS.LSReg
    fig, axs = plt.subplots(3,1)
    #fig.suptitle('Se)
    plt.tight_layout()

    Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xa)), np.squeeze(PS.Reg)
    varx = np.ma.masked_array(ndd[namex], mask = PS.totMask)
    #varx = PS.dimDict[namex]
    Regcolor = np.transpose(np.array([Reg, Reg]))
    Regcolor = np.ma.masked_array(Regcolor, mask = tm)
    #if namex == 'tau_w':
        #varx = invsymlog10(varx)
    

    yt = np.array([np.amin(Phi_0), np.amax(Phi_0)])
    x, y = np.meshgrid(varx, yt, indexing = 'ij')
    #x, y = x.ravel(), y.ravel()
    axs[0].plot(varx, Phi_0, lw = 2, c = 'k')
    axs[0].title.set_text(r'Mouth stratification $\Phi_0$')
    f1 = axs[0].contourf(x, y, Regcolor, cmap = plt.get_cmap('plasma'), alpha = 1)
    cb = plt.colorbar(f1, ax = axs[0], ticks = [1 ,2, 3, 4])
    
    
    yt = np.array([np.amin(Ls)/3, np.amax(Ls)*3])
    x, y = np.meshgrid(varx, yt, indexing = 'ij')
    axs[1].plot(varx, Ls, lw = 2, c = 'k', label = r'$L_s$')
    axs[1].title.set_text(r'Salt intrusion $L_s$')
    f2 = axs[1].contourf(x, y, Regcolor, cmap = plt.get_cmap('plasma'), alpha = 1)
    cb = plt.colorbar(f2, ax = axs[1], ticks = [1 ,2, 3, 4])
    
    LD = np.ma.masked_array(LSReg[6], mask = PS.totMask)
    LD = np.ma.masked_outside(LD, yt[0], yt[1])
    
    LGG = np.ma.masked_array(LSReg[0], mask = PS.totMask)
    LGG = np.ma.masked_outside(LGG, yt[0], yt[1])
    
    LWW = np.ma.masked_array(LSReg[5], mask = PS.totMask)
    LWW = np.ma.masked_outside(LWW, yt[0], yt[1])
    
    axs[1].plot(varx, LD, ls = '-', lw = 1, c = 'w', label = 'Dispersive (1)') #Dispersive Regime
    axs[1].plot(varx, LGG, ls = '--', lw = 1, c = 'w', label = 'Chatwin (2)') #Chatwin Regime
    axs[1].plot(varx, LWW, ls = '-.', lw = 1, c = 'w', label = 'Wind-driven (3-4)') #WW Regime
    #axs[1].plot(varx, Ls, lw = 2, c = 'k')
    
    axs[1].set_yscale('log')
    axs[1].legend()
    # Here, insert theoretical prediction.
    
    aTT = PS.aTT
    for ind in range(len(aTT)-1): axs[2].plot(varx, np.squeeze(aTT[ind]), color=col[ind], label = labT[ind])
    ind = len(aTT)-1
    axs[2].plot(varx, np.abs(np.squeeze(aTT[ind])), color=col[ind], label = labT[ind])
    axs[2].legend()
    axs[2].title.set_text('Transports')
    
    for i in range(3):
        if namex == 'Fw':
            axs[i].set_xscale('symlog')
        else:
            axs[i].set_xscale('log')
        axs[i].set_xlabel(namex)
        #axs[i].grid(True)

def plotNDim2(PS):
    ndd = PS.nondimDict
    namex, namey = ndd['pars']
    cmap = 'Blues'
    fig, axs = plt.subplots(3,1)
    #fig.suptitle('Sensitivity: Wind and Depth')
    #Ra, Fr, Fw, 
    Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xa)), np.squeeze(PS.Reg)
    varx = ndd[namex]
    vary = ndd[namey]
    plt.tight_layout()
    cfig = axs[0].contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Reg, 20, cmap = plt.get_cmap('plasma'))
    axs[0].contour(cfig, levels = np.linspace(.5, 4.5), colors = 'k', linewidths = 0.5)
    axs[0].title.set_text('Regime')
    cb = plt.colorbar(cfig, ax = axs[0], ticks = [1 ,2, 3, 4])

    hl = [np.min([np.amin(Phi_0),0]), np.max([np.amin(Phi_0),0]), np.min([np.amax(Phi_0),1]), np.max([np.amax(Phi_0),1])]
    hl.sort()
    minz, maxz = np.min(Phi_0), np.max(Phi_0)
    cfig = axs[1].contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Phi_0, 20, cmap = cmap, corner_mask = True, vmin = minz, vmax = maxz)
    axs[1].contour(cfig,  levels = np.linspace(minz, maxz), colors = 'k', linewidths = 0.5)
    axs[1].contour(cfig, levels = [0,1], colors='k', linewidths = 1)
    axs[1].contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Phi_0, levels = hl, vmin = minz, vmax = maxz, hatches = [".", "", "//"], alpha = 0)
    #axs[1].scatter(SM.Ra, SM.Fr, color = 'k', marker = 'o')    #axs[1,2].clabel(cfigc, cfigc.levels, inline = True, fontsize = 10)       
    axs[1].title.set_text(r'Stratification $\Phi_0$')
    plt.colorbar(cfig, ax = axs[1])

    minz, maxz = np.amin(Ls), np.amax(Ls)
    #print(np.amin(Ls))
    #cfig2 = axs[2,2].contourf(Ra, Fr, Ls, 20, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True, vmin = minz, vmax = maxz)
    cfig2 = axs[2].contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Ls, 50, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True)
    axs[2].contour(cfig2,  levels = np.geomspace(minz,maxz), colors = 'k', linewidths = 0.5)
    axs[2].contour(cfig2, levels = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000], colors='k', linewidths = 1)     
    axs[2].title.set_text(r'Salt intrusion $L_s$')
    plt.colorbar(cfig2, ax = axs[2])
    
    for i in range(3):
        if namex == 'Fw':
            axs[i].set_xscale('symlog')
            axs[i].set_xlabel('Fw')
            axs[i].plot([0,0], [np.amin(vary), np.amax(vary)], c = 'w') 
        else:
            axs[i].set_xscale('log')
            axs[i].set_xlabel(namex)
            
        if namey == 'Fw':
            axs[i].set_yscale('symlog')
            axs[i].set_ylabel('Fw')
        else:
            axs[i].set_yscale('log')
            axs[i].set_ylabel(namey) 

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
    

def plotDim1(PS, dimDict):
    col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    labT = ['G-G', 'G-R', 'G-W', 'R-R', 'R-W', 'W-W', 'DI', '|FL|']
    namex = dimDict['pars'][0]
    tm = np.transpose(np.array([PS.totMask, PS.totMask], dtype = bool))
    # Prepare background color for plots
    
    #Reg = ma.masked_array(Reg, mask = tm)
    
    LSReg = PS.LSReg
    fig, axs = plt.subplots(3,1)
    #fig.suptitle('Se)
    plt.tight_layout()

    Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xa)), np.squeeze(PS.Reg)
    varx = np.ma.masked_array(dimDict[namex], mask = PS.totMask)
    #varx = PS.dimDict[namex]
    Regcolor = np.transpose(np.array([Reg, Reg]))
    Regcolor = np.ma.masked_array(Regcolor, mask = tm)
    #if namex == 'tau_w':
        #varx = invsymlog10(varx)
    

    yt = np.array([np.amin(Phi_0), np.amax(Phi_0)])
    x, y = np.meshgrid(varx, yt, indexing = 'ij')
    #x, y = x.ravel(), y.ravel()
    axs[0].plot(varx, Phi_0, lw = 2, c = 'k')
    axs[0].title.set_text(r'Mouth stratification $\Phi_0$')
    f1 = axs[0].contourf(x, y, Regcolor, cmap = plt.get_cmap('plasma'), alpha = 1)
    cb = plt.colorbar(f1, ax = axs[0], ticks = [1 ,2, 3, 4])
    
    
    yt = np.array([np.amin(Ls)/3, np.amax(Ls)*3])
    x, y = np.meshgrid(varx, yt, indexing = 'ij')
    axs[1].plot(varx, Ls, lw = 2, c = 'k', label = r'$L_s$')
    axs[1].title.set_text(r'Salt intrusion $L_s$')
    f2 = axs[1].contourf(x, y, Regcolor, cmap = plt.get_cmap('plasma'), alpha = 1)
    cb = plt.colorbar(f2, ax = axs[1], ticks = [1 ,2, 3, 4])
    
    LD = np.ma.masked_array(LSReg[6], mask = PS.totMask)
    LD = np.ma.masked_outside(LD, yt[0], yt[1])
    
    LGG = np.ma.masked_array(LSReg[0], mask = PS.totMask)
    LGG = np.ma.masked_outside(LGG, yt[0], yt[1])
    
    LWW = np.ma.masked_array(LSReg[5], mask = PS.totMask)
    LWW = np.ma.masked_outside(LWW, yt[0], yt[1])
    
    axs[1].plot(varx, LD, ls = '-', lw = 1, c = 'w', label = 'Dispersive (1)') #Dispersive Regime
    axs[1].plot(varx, LGG, ls = '--', lw = 1, c = 'w', label = 'Chatwin (2)') #Chatwin Regime
    axs[1].plot(varx, LWW, ls = '-.', lw = 1, c = 'w', label = 'Wind-driven (3-4)') #WW Regime
    #axs[1].plot(varx, Ls, lw = 2, c = 'k')
    
    axs[1].set_yscale('log')
    axs[1].legend()
    # Here, insert theoretical prediction.
    
    aTT = PS.aTT
    for ind in range(len(aTT)-1): axs[2].plot(varx, np.squeeze(aTT[ind]), color=col[ind], label = labT[ind])
    ind = len(aTT)-1
    axs[2].plot(varx, np.abs(np.squeeze(aTT[ind])), color=col[ind], label = labT[ind])
    axs[2].legend()
    axs[2].title.set_text('Transports')
    
    for i in range(3):
        if namex == 'H':
            axs[i].set_xscale('log')
        elif namex == 'tau_w':
            axs[i].set_xscale('symlog')
        else:
            axs[i].set_xscale('log')
        axs[i].set_xlabel(namex)
        #axs[i].grid(True)

    
def plotDim2(PS, dimDict):
    namex, namey = dimDict['pars']
    cmap = 'Blues'
    fig, axs = plt.subplots(3,1)
    #fig.suptitle('Sensitivity: Wind and Depth')
    #Ra, Fr, Fw, 
    Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xa)), np.squeeze(PS.Reg)
    varx = dimDict[namex]
    vary = dimDict[namey]
    plt.tight_layout()
    cfig = axs[0].contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Reg, 20, cmap = plt.get_cmap('plasma'))
    axs[0].contour(cfig, levels = np.linspace(.5, 4.5), colors = 'k', linewidths = 0.5)
    #axs[0,2].contour(cfig, levels = [0,1], colors='k', linewidths = 1)
    #axs[0,2].contourf(Ra, Fr, Reg, levels = hl, vmin = minz, vmax = maxz, hatches = [".", "", "//"], alpha = 0)
    #axs[0,2].scatter(SM.Ra, SM.Fr, color = 'k', marker = 'o')    #axs[1,2].clabel(cfigc, cfigc.levels, inline = True, fontsize = 10)       
    axs[0].title.set_text('Regime')
    cb = plt.colorbar(cfig, ax = axs[0], ticks = [1 ,2, 3, 4])
    #cb.ax.set_yticklabels(['1', '2', '3', '4'])
    #cb = plt.colorbar(cfig, ax = axs[0,2])
    #cb.ax.set_yticklabels(['1', '2', '3', '4'])    

    hl = [np.min([np.amin(Phi_0),0]), np.max([np.amin(Phi_0),0]), np.min([np.amax(Phi_0),1]), np.max([np.amax(Phi_0),1])]
    hl.sort()
    minz, maxz = np.min(Phi_0), np.max(Phi_0)
    cfig = axs[1].contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Phi_0, 20, cmap = cmap, corner_mask = True, vmin = minz, vmax = maxz)
    axs[1].contour(cfig,  levels = np.linspace(minz, maxz), colors = 'k', linewidths = 0.5)
    axs[1].contour(cfig, levels = [0,1], colors='k', linewidths = 1)
    axs[1].contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Phi_0, levels = hl, vmin = minz, vmax = maxz, hatches = [".", "", "//"], alpha = 0)
    #axs[1].scatter(SM.Ra, SM.Fr, color = 'k', marker = 'o')    #axs[1,2].clabel(cfigc, cfigc.levels, inline = True, fontsize = 10)       
    axs[1].title.set_text(r'Stratification $\Phi_0$')
    plt.colorbar(cfig, ax = axs[1])

    minz, maxz = np.amin(Ls), np.amax(Ls)
    #print(np.amin(Ls))
    #cfig2 = axs[2,2].contourf(Ra, Fr, Ls, 20, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True, vmin = minz, vmax = maxz)
    cfig2 = axs[2].contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Ls, 50, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True)
    axs[2].contour(cfig2,  levels = np.geomspace(minz,maxz), colors = 'k', linewidths = 0.5)
    axs[2].contour(cfig2, levels = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000], colors='k', linewidths = 1)     
    axs[2].title.set_text(r'Salt intrusion $L_s$')
    plt.colorbar(cfig2, ax = axs[2])
    
    for i in range(3):
        if namex == 'tau_w':
            axs[i].set_xscale('symlog')
            axs[i].set_xlabel(r'$\tau_w$')
            axs[i].plot([0,0], [np.amin(vary), np.amax(vary)], c = 'w') 
        else:
            axs[i].set_xscale('log')
            axs[i].set_xlabel(namex)
            
        if namey == 'tau_w':
            axs[i].set_yscale('symlog')
            axs[i].set_ylabel(r'$\tau_w$')
        else:
            axs[i].set_yscale('log')
            axs[i].set_ylabel(namey) 


def plotDim3(PS, dimDict):
    names = dimDict['pars'] #tuple of variables that were varied.
    pars = []
    for i in range(3):
        if names[i] == 'tau_w':
            pars.append(symlog10(dimDict[names[i]]))
        else:
            pars.append(np.log10(dimDict[names[i]]))
            
    cmap = 'Blues'
    
    Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xa)), np.squeeze(PS.Reg)
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
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    f4 = ax.scatter(sRa, sFr, sFw, c = PS.Sb_X0.ravel(), cmap = 'Blues') #np.log10(PS.Ls.ravel())
    plt.colorbar(f4, ax = ax)
    ax.set_xlabel('Ra')
    ax.set_ylabel('Fr')
    ax.set_zlabel('Fw')
    ax.set_title(r'Salinity gradient $\bar{\Sigma}_{X0}$')
    '''
    
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
    axs[1,0].contour(f1, levels = np.linspace(0,1,50), colors = 'k', linewidths = 0.5, alpha = 0.2)
    axs[1,0].title.set_text(f'Salinity: Negative = {SM.maskNEG} and Unstable = {SM.maskIS}')
    axs[1,0].set_xlabel(r'$X$')
    axs[1,0].set_ylabel(r'$\sigma$')
    if np.amin(S) < 0:
        axs[1,0].contourf(Xp, sigmap, S, levels = [np.amin(S), 0, np.amax(S)], cmap = cmap, corner_mask = True, hatches=["//", ""], alpha = 0)
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
    
    Ra, Fr, Fw, Phi_0, Ls, Reg = np.squeeze(PS.Ra), np.squeeze(PS.Fr), np.squeeze(PS.Fw), np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xa)), np.squeeze(PS.Reg)
    
    cfig = axs[0,2].contourf(Ra, Fr, Reg, 20, cmap = plt.get_cmap('plasma'))
    #axs[0,2].contour(cfig,  colors = 'k', linewidths = 0.5)
    #axs[0,2].contour(cfig, levels = [0,1], colors='k', linewidths = 1)
    #axs[0,2].contourf(Ra, Fr, Reg, levels = hl, vmin = minz, vmax = maxz, hatches = [".", "", "//"], alpha = 0)
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
    axs[1,2].contourf(Ra, Fr, Phi_0, levels = hl, vmin = minz, vmax = maxz, hatches = [".", "", "//"], alpha = 0)
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
    #axs[1,2].contourf(Ra, Fr, Xa, levels = [np.min([np.amin(Xa),-1]), np.max([np.amin(self.Xa),-1]),0], vmin = minz, vmax = maxz, hatches = [".", "", "//"], alpha = 0)
    axs[2,2].scatter(SM.Ra, SM.Fr, color = 'k', marker = 'o')    #axs[1,2].clabel(cfigc, cfigc.levels, inline = True, fontsize = 10)       
    axs[2,2].title.set_text(r'Salt Intrusion $L_s$')
    axs[2,2].set_xlabel(r'Ra')
    axs[2,2].set_ylabel(r'Fr')
    axs[2,2].set_xscale('log')
    axs[2,2].set_yscale('log')
    plt.colorbar(cfig2, format = ticker.LogFormatterMathtext(), ax = axs[2,2])
    
def plotSModel(SM):
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
    axs[1,0].contour(f1, levels = np.linspace(0,1,50), colors = 'k', linewidths = 0.5, alpha = 0.2)
    axs[1,0].title.set_text(f'Salinity: Negative = {SM.maskNEG} and Unstable = {SM.maskIS}')
    axs[1,0].set_xlabel(r'$X$')
    axs[1,0].set_ylabel(r'$\sigma$')
    if np.amin(S) < 0:
        axs[1,0].contourf(Xp, sigmap, S, levels = [np.amin(S), 0, np.amax(S)], cmap = cmap, corner_mask = True, hatches=["//", ""], alpha = 0)
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
    axs[1,1].title.set_text('Integrated salt transport: Regime = '+ np.array2string(*SM.Reg))
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
    
    
def addC(BC):
    C = np.zeros(20)
    if BC == 0:
        C[0] = 0 #dummy
        C[1] = 493/362880 #-P2P5
        C[2] = 0 #-P1P5 = -P2P4
        C[3] = 2*-197/120960 #-P3P5 = -P2P6
        C[4] = 0 #-P1P4
        C[5] = 0 #-P1P6 = -P3P4
        C[6] = 2/945 #-P3P6
        C[7] = C[2]*C[1]**(-2/3)
        C[8] = C[3]*C[1]**(-2/3)
        C[9] = C[4]*C[1]**(-1/3)
        C[10] = C[5]*C[1]**(-1/3)
        C[11] = C[6]*C[1]**(-1/3)
        C[12] = C[1]**(-1/3)
        C[13] = C[1]**(-2/3)

        C[14] = 0 #P4(0)
        C[15] = 11/720 #P5(0)
        C[16] = -1/45 #P6(0)
        C[17] = 0 #P4(-1)
        C[18] = -13/720 #P5(-1)
        C[19] = 7/360 #P6(-1)
        
    if BC == 1:
        C[0] = 0 #dummy
        C[1] = 881/18144000 #-P2P5
        C[2] = 2*191/504000 #-P1P5 = -P2P4
        C[3] = 2*43/168000 #-P3P5 = -P2P6
        C[4] = 8/2625 #-P1P4
        C[5] = 2*41/21000 #-P1P6 = -P3P4
        C[6] = 29/21000 #-P3P6
        C[7] = C[2]*C[1]**(-2/3)
        C[8] = C[3]*C[1]**(-2/3)
        C[9] = C[4]*C[1]**(-1/3)
        C[10] = C[5]*C[1]**(-1/3)
        C[11] = C[6]*C[1]**(-1/3)
        C[12] = C[1]**(-1/3)
        C[13] = C[1]**(-2/3)

        C[14] = -7/300 #P4(0)
        C[15] = -23/7200 #P5(0)
        C[16] = -11/600 #P6(0)
        C[17] = 2/75 #P4(-1)
        C[18] = 11/3600  #P5(-1)
        C[19] = 3/200 #P6(-1)
    
    if BC == -1:
        C[0] = 0 #dummy
        C[1] = 19/1451520 #P2P5
        C[2] = 2*19/40320 #P1P5 = P2P4
        C[3] = 2*1/11520 #P3P5 = P2P6
        C[4] = 2/105 #P1P4
        C[5] = 2*1/336 #P1P6 = P3P4
        C[6] = 1/1680 #P3P6
        C[7] = C[2]*C[1]**(-2/3)
        C[8] = C[3]*C[1]**(-2/3)
        C[9] = C[4]*C[1]**(-1/3)
        C[10] = C[5]*C[1]**(-1/3)
        C[11] = C[6]*C[1]**(-1/3)
        C[12] = C[1]**(-1/3)
        C[13] = C[1]**(-2/3)

        C[14] = -7/120 #P4(0)
        C[15] = -1/576 #P5(0)
        C[16] = -1/80 #P6(0)
        C[17] = 1/15 #P4(-1)
        C[18] = 1/720 #P5(-1)
        C[19] = 1/120 #P6(-1)
    #Sc = 2.2
    return C

def formFunctions(sigmap, BC):
    if BC == 1:
        P1 = 1/5-3/5*sigmap**2
        P2 = 1/30 - 9/40*sigmap**2 - 1/6*sigmap**3
        P3 = 3/10 + sigmap + 3/5*sigmap**2
        P4 = -7/300 + sigmap**2/10 - sigmap**4/20
        P5 = -23/7200 + 1/60*sigmap**2 - 3/160*sigmap**4 - 1/120*sigmap**5
        P6 = -11/600 + 3/20*sigmap**2 + sigmap**3/6 + sigmap**4/20 
        P7 = 1/30*sigmap - 3/40*sigmap**3 - 1/24*sigmap**4
        
    if BC == 0:
        P1 = 0*sigmap
        P2 = -1/6*sigmap**3 + sigmap**2/4 - 1/8
        P3 = sigmap**2/2 + sigmap + 1/3
        P4 = 0*sigmap
        P5 = -sigmap**5/120 + sigmap**4/48 - sigmap**2/16 + 11/720
        P6 = sigmap**4/24 + sigmap**3/6 + sigmap**2/6 - 1/45
        P7 = -1/8*sigmap - 1/24*sigmap**4 + 1/12*sigmap**3

    if BC == -1:
        P1 = -3/2*sigmap**2 + 1/2
        P2 = -sigmap**3/6 - 3/16*sigmap**2 + 1/48
        P3 = 3/4*sigmap**2 + sigmap + 1/4
        P4 = -1/8*sigmap**4 + sigmap**2/4 - 7/120
        P5 = -sigmap**5/120 - sigmap**4/64 + sigmap**2/96 - 1/576
        P6 = sigmap**4/16 + sigmap**3/6 + sigmap**2/8 - 1/80
        P7 = 1/48*sigmap - 1/24*sigmap**4 - 1/16*sigmap**3
    
    return P1, P2, P3, P4, P5, P6, P7

#integral functions for computing transport terms