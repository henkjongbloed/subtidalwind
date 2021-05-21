import numpy as np
from numpy.linalg.linalg import qr
from numpy.ma.extras import masked_all
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from matplotlib import ticker, cm
import matplotlib.tri as mtri
import matplotlib.colors as co

def globalParameters(**kwargs):
    gp = dict(R = 2, #R: real or Inf
        q = 1/30, #Salinity fraction at landward end
        n = [1, 401, 31, 13], #0 parameters to vary, 1 parameter, 2 parameters, 3 parameters.
        SM = [1001,21], #Single model run: grid.
        PS = [51,11],
        tolNEG = .1,
        tolUN = .1,
        tolPrit = 10, #not employed in current version of script.
        Sc = 2.2,
        m = 1.2, #tuning factor for mixing algorithm
        Ori = True, #Plots 'Original' parameters instead of modified ones. (now; only NonDim compatible)
        )
    
    gp['C'] = addC(gp['R'])
    for key, value in kwargs.items(): #Change the constant parameters
            gp[key] = value
    return gp

def solveCubic(r):
    '''Computes single real solution based on parameter values'''
    solC = np.roots(r)
    D3 = computeD3(r[0],r[1],r[2],r[3])
    mask = (D3 >= 0)
    if ~mask: #one real solution
        sol = float(np.real(solC[np.isreal(solC)]))
        return sol, mask
    else:
        return mask, mask

def computeD3(a,b,c,d):
    D3 = b**2*c**2 - 4*a*c**3 - 4*b**3*d - 27*a**2*d**2 + 18*a*b*c*d
    return D3

def computeD2(a,b,c):
    D2 = b**2 - 4*a*c
    return D2

def Ra2Ft(Ra):
    '''
    Using the Banas and Ralston parametrisations
    '''
    return Ra**(-1/2)/(3.9e5)

def Ft2Ra(Ft):
    '''
    Using the Banas and Ralston parametrisations
    '''
    return 3.9e5*Ft**-2

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
    return D2, np.array([bpx, xright, xleft]), np.array([bpy, yright, yleft])

def computeLocalExtremaVect(a,b,c,d):
    '''Returns the discriminant and local extrema of the S - S_X curve'''
    D2 = computeD2(3*a, 2*b, c)
    bpx = -b/(3*a) #x where second derivative is zero
    bpy = a*bpx**3 + b*bpx**2 + c*bpx + d #y where second derivative is zero
    xright, xleft, yright, yleft = bpx, bpx, bpy, bpy #no local extrema
    xright[D2>=0] = (-2*b[D2>=0] + np.sqrt(D2[D2>=0]))/(6*a[D2>=0])#x where first derivative is zero
    xleft[D2>=0] = (-2*b[D2>=0] - np.sqrt(D2[D2>=0]))/(6*a[D2>=0])#x where first derivative is zero
    yright[D2>=0] = a[D2>=0]*xright[D2>=0]**3 + b[D2>=0]*xright[D2>=0]**2 + c[D2>=0]*xright[D2>=0] + d[D2>=0]#y where first derivative is zero
    yleft[D2>=0] = a[D2>=0]*xleft[D2>=0]**3 + b[D2>=0]*xleft[D2>=0]**2 + c[D2>=0]*xleft[D2>=0] + d[D2>=0]#y where first derivative is zero
    return D2, np.transpose(np.array([bpx, xright, xleft])), np.transpose(np.array([bpy, yright, yleft]))

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

def defaultDim(gp):
    dd = dict(Q = 1000, 
        H = 20,  
        K_M = 1e-3, 
        tau_w = 0,
        K_H = 250, 
        s_0 = 30.0, g = 9.81, beta = 7.6e-4, rho_0 = 1000.0, B = 1000.0, Sc = gp['Sc'])
    return dd

def makeDicts(gp, *args, **kwargs):
    #First default values by Dijkstra and Schuttelaars (2021)
    dd = defaultDim(gp)
    
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
    H = np.logspace(0,2.5,n)
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
    if len(ndd['pars']) > 1:
        n = np.prod(ndd['nps'])
    else:
        n = ndd['nps']
    ndd['Ra'], ndd['Fr'], ndd['Fw'] = ndd['Ra']*np.ones(n), ndd['Fr']*np.ones(n), ndd['Fw']*np.ones(n) #Make sure everything is the same size.
    ndd['n'] = n
    
    if 'name' in kwargs:
        ndd['name'] = kwargs['name']
    else:
        ndd['name'] = ndd['pars']
        dd['name'] = dd['pars']
        
    dd['c'] = np.sqrt(dd['g']*dd['beta']*dd['s_0']*dd['H'])
    
    return dd, ndd

def makeNDDict(gp, *args, **kwargs):
    dd = defaultDim(gp)
    ndd = dim2nondim(dd) #Default, also for nonDim model run.
    
    for key, value in kwargs.items(): #Change the constant parameters
        if key not in ['Ralston', 'Sverdrup']:
            ndd[key] = value
        
    if len(args) == 0:
        #ndd = dim2nondim(dd)
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
    Ra = np.logspace(1, 5, n)    
    Fr = np.logspace(-4 , 0 , n)
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

    #ndd = dim2nondim(dd)s
    ndd['pars'] = args
    ndd['Sc'] = gp['Sc']

    #Ra, Fr, Fw = np.meshgrid(Ra1, Fr1, Fw1, indexing='ij')
    #nondimDict = dict(Ra = Ra.flatten(),Fr = Fr.flatten(), Fw = Fw.flatten(), Sc = dd['Sc'], nps = nps)
    
    if len(ndd['pars']) > 1:
        n = np.prod(ndd['nps'])
    else:
        n = ndd['nps']
    ndd['Ra'], ndd['Fr'], ndd['Fw'] = ndd['Ra']*np.ones(n), ndd['Fr']*np.ones(n), ndd['Fw']*np.ones(n) #Make sure everything is the same size.
    ndd['n'] = n
    return ndd

def Ralston(u_T, H, B):
    K_M = 0.028*0.0026*u_T*H
    K_H = 0.035*u_T*B
    return K_M, K_H

def Sverdrup(tau_w, K_M0, beta):
    K_M = K_M0 + beta*np.abs(tau_w)
    return K_M

def computersXs(gp, a, b, c, d, Sb_X0):
    '''
    
    Computes rs and Xs (salt intrusion coordinates)
    
    Input: ODE data
    
    Output: rs (float), Xs (float), mask (bool)
    
    '''
    q = gp['q']
    expr, mask = solveCubic(np.array([a*Sb_X0**2, b*Sb_X0, c, -q*(a*Sb_X0**2 + b*Sb_X0 + c)])) #rt = exp(r)
    if (expr < 1.0) and ~mask:
        rs = np.log(expr)
        Xs = (3/2*a*Sb_X0**2*(np.exp(2*rs)-1) + 2*b*Sb_X0*(np.exp(rs)-1) + c*rs)/d
    else:
        mask = True
        return mask, mask, mask
    return rs, Xs, mask

def scaledTransport(L, Sb, Sb_X):
    ''' Computes scaled (divided by d*Sigma) transport.
    
    Input: ODE and salt profiles
    
    Output: TX (List of size 8 consisting of numpy arrays of shape Grid[0])
    
    '''
    P = [Sb_X**3/Sb,    #GG
        Sb_X**2/Sb,     #GR
        Sb_X**2/Sb,     #GW
        Sb_X/Sb,#RR
        Sb_X/Sb,#RW
        Sb_X/Sb,#WW
        Sb_X/Sb,#D
        Sb/Sb#F
        ] #For numerical integration.
    
    TX = [l*p/L[7] for l,p in zip(L, P)] #transports as function of X, scaled by the river lengtscale (Fr)
    return TX

def scaledIntegratedTransport(gp, L, a, b, c, Sb_X0, rs, Xs):
    ''' Computes integrated (Xs to 0), divided by abs(Xs) and scaled (divided by d*Sigma) transport.
    
    Input: ODE and salt intrusion data
    
    Output: T (numpy array of shape 8)
    
    '''
    q = gp['q']
    
    bl, bu = Sb_X0*np.exp(rs), Sb_X0
    
    sT = [Int3(bl, bu, a, b, c), #Analytically integrated P
            Int2(bl, bu, a, b, c),
            Int2(bl, bu, a, b, c),
            -np.log(q), #L[0] is just a dummy variable, no length
            -np.log(q),
            -np.log(q),
            -np.log(q),
            -Xs
            ]
    
    T = np.array([l*st/(-Xs*L[7]) for l, st in zip(L, sT)]) #analytical integration, scaled
    return T

def solveODE(gp, a, b, c, d, Sb_X0, rs):
    '''
    
    Solves the qubic ODE.
    
    Output: r, X, Sb, Sb_X, Sb_XX (all numpy vectors)
    
    '''
    #C = gp['C']
    r = np.linspace(rs, 0, gp[gp['Class']][0])
    X = (3/2*Sb_X0**2*a*(np.exp(2*r)-1) + 2*b*Sb_X0*(np.exp(r)-1) + c*r)/d
    Sb = (a*Sb_X0**3*np.exp(3*r) + b*Sb_X0**2*np.exp(2*r) + c*Sb_X0*np.exp(r))/d
    Sb_X = Sb_X0*np.exp(r)
    Sb_XX = d*Sb_X/(3*a*Sb_X**2 + 2*b*Sb_X + c)
    return r, X, Sb, Sb_X, Sb_XX

def nonDim2ODECoeff(gp, Ra, Fr, Fw, Sc):
    '''Input: Nondimensional Parameters
    
    Output: Length scales, ODE factors, mouth factors. (np arrays, same dim as input parameters)
    '''
    C = gp['C']
    L = np.transpose(np.array([C[0]*Sc*Ra**3,                     #GG cubed
        C[1]*Sc*Ra**2*Fr,                   #GR squared
        C[2]*Sc*Ra**2*Fw,                   #GW squared
        C[3]*Sc*Ra*Fr**2,                   #RR
        C[4]*Sc*Ra*Fr*Fw,                   #RW
        C[5]*Sc*Ra*Fw**2,                   #WW
        np.ones_like(Ra),                                #D
        Fr]))                             #F
    
    #if gp['Class'] == 'PS':
    #    a = L[:,0]
    #    b = L[:,1] + L[:,2]
    #    c = L[:,3] + L[:,4] + L[:,5] + L[:,6]
    #    d = L[:,7]
    #else:
    a = L[0]
    b = L[1] + L[2]
    c = L[3] + L[4] + L[5] + L[6]
    d = L[7]
        
    b0 = b + d*Sc*Ra**2*C[10]
    c0 = c + d*Sc*Ra*(Fr*C[9] + Fw*C[11])
    return L, a, b, c, d, b0, c0

def processBC(gp, a, b, c, d, Ra, Fr, Fw, Sc, Sb_X0):
    C = gp['C']
    Sb_0 = (a*Sb_X0**3 + b*Sb_X0**2 + c*Sb_X0) / d
    S_00 = Sb_0 + Sc*Ra*Sb_X0*(Fr*C[6] + Ra*Sb_X0*C[7] + Fw*C[8])
    Phi_0 = 1 - S_00
    return Sb_0, Phi_0

def getGrid(gp, X):
    '''
    Computes meshgrid of X and sigma
    
    Input: gp and GridDim (either 'SMGrid' or 'PsGrid')
    
    Output: Xp, sigmap (2d arrays of dimension GridDim) and sigma (1d array of dimension GridDim[1])
    '''
    
    sigma = np.linspace(-1,0,gp[gp['Class']][1])
    Xp, sigmap = np.meshgrid(X,sigma)
    SF = formFunctions(sigmap, gp['R'])
    return Xp, sigmap, sigma, SF

def computeS(gp, Ra, Fr, Fw, Sc, P, Sb, Sb_X, Sb_XX):
    '''
    Input: Avg salinity, derivatives and other parameters. 
    
    Output: S, S_X, Sbar, Sacc, Sbar_X, Sacc_X
    
    '''
    nsigma = gp[gp['Class']][1]
    Sbar, Sbar_X = np.matlib.repmat(np.transpose(Sb),nsigma,1), np.matlib.repmat(np.transpose(Sb_X),nsigma,1)
    Sacc, Sacc_X = Sc*Ra*(Fr*P[3] + Fw*P[5])*np.matlib.repmat(np.transpose(Sb_X),nsigma,1) + Sc*Ra**2*P[4]*np.matlib.repmat(np.transpose(Sb_X),nsigma,1)**2, Sc*Ra*(Fr*P[3] + Fw*P[5])*np.matlib.repmat(np.transpose(Sb_XX),nsigma,1) + Sc*Ra**2*P[4]*np.matlib.repmat(np.transpose(2*Sb_X*Sb_XX),nsigma,1)
    S =  Sbar + Sacc
    S_X =  Sbar_X, Sacc_X
    return S, S_X, Sbar, Sacc, Sbar_X, Sacc_X

def computeSurfBottom(gp, Ra, Fr, Fw, Sc, Sb, Sb_X):
    '''Input: Avg salinity, derivatives and other parameters. Output: S_surface and S_bottom and components.'''
    C = gp['C']
    Ssurf = Sb + Sc*Ra*(Fr*C[6] + Fw*C[8])*Sb_X + Sc*Ra**2*C[7]*Sb_X**2
    Sbot = Sb + Sc*Ra*(Fr*C[9] + Fw*C[11])*Sb_X + Sc*Ra**2*C[10]*Sb_X**2
    return Ssurf, Sbot

def computeU(gp, Ra, Fr, Fw, P, Sb_X, Sb_XX):
    '''
    Input: Avg salinity gradient and other parameters
    
    Output: U, W, UR, UG, UW (all numpy arrays)
    
    '''
    nsigma = gp[gp['Class']][1]
    Ubar = Fr*np.ones_like(P[0])    
    UR = Fr*P[0]
    UG = Ra*P[1]*np.matlib.repmat(np.transpose(Sb_X),nsigma,1)
    UW = Fw*P[2]   
    #temp = np.transpose(Sb)# np.matlib.repmat(np.transpose(Sb),nsigma,1) -> Start here!!!!
    #print(temp)
    U =  Ubar + UR + UG + UW
    W = -P[6]*np.matlib.repmat(np.transpose(Sb_XX),nsigma,1) # In this, we have scaled with K_M/c * Ra**2
    return U, W, Ubar, UR, UG, UW

def computeNU(gp, D2, Exx, a, b, c, d, Sb_X0, Sb_0):
    maskNU = False
    if D2 >=0:
        Sb_Xs, _ = solveCubic(np.array([a, b, c, -d*gp['q']*Sb_0]))
        if (Sb_Xs <= Exx[0] <= Sb_X0) or (Sb_Xs <= Exx[1] <= Sb_X0):
            maskNU = True
    return maskNU

def computePhysicalMasks(gp, Ra, Fr, Fw, Sc, Sb, Sb_X):
    Ssurf, Sbot = computeSurfBottom(gp, Ra, Fr, Fw, Sc, Sb, Sb_X)
    maskNEG = np.any((Ssurf < -gp['tolNEG'])) # 1 if there is negative salt (within tolNEG)
    maskUN = np.any((Sbot - Ssurf < -gp['tolUN'])) # 1 if there exists a point of unstable stratification (within tolUNSTABLE)
    return maskNEG, maskUN

def computePhysicalMasksPS(gp, a, b, c, d, Ra, Fr, Fw, Sc, Sb_X0, rs):
    '''
    Returns maskNEG and maskUNSTABLE (tolerances specified by gp)
    
    Only looks at surface and bottom salinity, not intermediate.
    
    '''
    _,_, Sb, Sb_X, _ = solveODE(gp, a, b, c, d, Sb_X0, rs)
    Ssurf, Sbot = computeSurfBottom(gp, Ra, Fr, Fw, Sc, Sb, Sb_X)
    maskNEG = np.any((Ssurf < -gp['tolNEG'])) # 1 if there is negative salt
    maskUN = np.any((Sbot - Ssurf < -gp['tolUN'])) # 1 if there exists a point of unstable stratification
    return maskNEG, maskUN


def findMixing(fac, Ra0, Fw0):
    '''
    Increases K_M in case any mask equals 1. This is equivalent to decreasing Ra and Fw by the same factor.
    '''
    Ra = Ra0/fac
    Fw = Fw0/fac
    #print('fix')
    return Ra, Fw, True

#def computeScaling()

def Int3(bl, bu, a, b, c):
    return 3*a*(I3(bu,a,b,c)-I3(bl,a,b,c)) + 2*b*(I2(bu,a,b,c)-I2(bl,a,b,c)) + c*(I1(bu,a,b,c)-I1(bl,a,b,c))

def Int2(bl, bu, a, b, c):
    return 3*a*(I2(bu,a,b,c)-I2(bl,a,b,c)) + 2*b*(I1(bu,a,b,c)-I1(bl,a,b,c)) + c*(I0(bu,a,b,c)-I0(bl,a,b,c))

def I0(x, a, b, c):
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

def I1(x, a, b, c):
    ''' Compute the integral of u/(au^2 + bu + c) and fill in x'''
    I = 1/(2*a)*np.log(np.abs(a*x**2 + b*x + c)) - b/(2*a)*I0(x,a,b,c)
    return I

def I2(x, a, b, c):
    ''' Compute the integral of u^2/(au^2 + bu + c) and fill in x'''
    I = x/a - b/(2*a**2)*np.log(np.abs(a*x**2 + b*x + c)) + (b**2 - 2*a*c)/(2*a**2)*I0(x,a,b,c)
    return I

def I3(x, a, b, c):
    ''' Compute the integral of u^3/(au^2 + bu + c) and fill in x'''
    I = x**2/(2*a) - b*x/a**2 + (b**2-a*c)/(2*a**3)*np.log(np.abs(a*x**2 + b*x + c)) - b*(b**2 - 3*a*c)/(2*a**3)*I0(x,a,b,c)
    return I

def computeScore(T):
    '''
    Computes scores of Regime 1 (Dispersive), Regime 2 (Chatwin) and 2 (Wind-Driven)
    '''
    s = np.array([T[6], T[0]+T[1]+T[3], abs(T[2]) + abs(T[4]) + abs(T[5])])/np.sum(np.abs(T[0:7]))
    #maxT = np.argmax(np.abs(T[0:7]))
    #alpha = 1.0 - np.sum(s[~np.argmax(s)])
    return s

def computeRegime(T):
    '''
    
    Input: Integrated transport vector
    
    Output: RBG regime vector.
    
    '''
    s = computeScore(T)
    regMat = np.array([[1,0,0], [0,1,0], [0,0,1]])
    r = np.matmul(regMat,s)
    return r
    
def computeLsTheory(gp,L,Sb_0):
    q = gp['q']
    LsT = np.array([3/2*Sb_0**(2/3)*L[0]**(1/3)/L[7]**(1/3)*(1-q**2/3),
                2*np.sqrt(L[1])*(np.sqrt(Sb_0) - q**2*Sb_0**2)/L[7], 
                np.sign(L[2])*2*np.sqrt(np.abs(L[2]))*(np.sqrt(Sb_0) - q**2*Sb_0**2)/L[7],
                -L[3]/L[7]*np.log(Sb_0*q),
                -L[4]/L[7]*np.log(Sb_0*q),
                -L[5]/L[7]*np.log(Sb_0*q),
                -L[6]/L[7]*np.log(Sb_0*q)]
        )
    return LsT
    
    
#def initMixPar(gp):
    #mixPar = dict()
def symlog10(x,l = 1/np.log(10)):
    return np.sign(x)*np.log10(1.0 + np.abs(x/l))
    
def invsymlog10(y,l = 1/np.log(10)):
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
    
    LD = np.ma.masked_array(LsT[:,6])
    LD = np.ma.masked_outside(LD, Ll, Lu)
    
    LGG = np.ma.masked_array(LsT[:,0])
    LGG = np.ma.masked_outside(LGG, Ll, Lu)
    
    LWW = np.ma.masked_array(LsT[:,5])
    LWW = np.ma.masked_outside(LWW, Ll, Lu)
    fig, axs = plt.subplots(3,1, sharex = True)

    #plt.tight_layout()

    plt.suptitle(PS.name)    
    f1 = axs[0].imshow(Regcolor, extent = extentp, origin = 'upper', aspect = 'auto')
    axs[0].plot(varx, Phi_0, lw = 2, c = 'k')
    axs[0].contourf(*np.meshgrid(varx, np.array([np.amin(Phi_0), np.amax(Phi_0)]), indexing = 'ij'), np.transpose(np.array([PS.mask[:,5], PS.mask[:,5]])), 1, hatches = [" ", "//"], alpha = 0)
    axs[0].contourf(*np.meshgrid(varx, np.array([np.amin(Phi_0), np.amax(Phi_0)]), indexing = 'ij'), np.transpose(np.array([PS.mask[:,6], PS.mask[:,6]])), 1, hatches = [" ", "."], alpha = 0)
    axs[0].title.set_text(r'Mouth stratification $\Phi_0$')
    if 'Fw' in namex:
        axs[0].plot([0,0], np.array([np.amin(Phi_0), np.amax(Phi_0)]), c = 'w')

    axs[1].plot(varx, Ls, lw = 2, c = 'k', label = r'$L_s$')
    axs[1].title.set_text(r'Salt intrusion $\Lambda_s$ (NonDim)')
    f2 = axs[1].imshow(Regcolor, extent = extentl, origin = 'upper', aspect = 'auto')
    axs[1].plot(varx, LD, ls = '-', lw = 1, c = 'w', label = 'Dispersive') #Dispersive Regime
    axs[1].plot(varx, LGG, ls = '--', lw = 1, c = 'w', label = 'Chatwin') #Chatwin Regime
    axs[1].plot(varx, LWW, ls = '-.', lw = 1, c = 'w', label = 'Wind-driven') #WW Regime
    axs[1].contourf(*np.meshgrid(varx, np.array([Ll, Lu]), indexing = 'ij'), np.transpose(np.array([PS.mask[:,5], PS.mask[:,5]])), 1, hatches = [" ", "//"], alpha = 0)
    axs[1].contourf(*np.meshgrid(varx, np.array([Ll, Lu]), indexing = 'ij'), np.transpose(np.array([PS.mask[:,6], PS.mask[:,6]])), 1, hatches = [" ", "."], alpha = 0)
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
    
    #for i in range(3):
        #axs[i].set_ylabel(ndd['pars'][1])
        #if 'Fw' in namex:
            #axs[i].plot([0,0], [np.amin(vary), np.amax(vary)], c = 'w') 
    
    #if 'tau_w' in namex:
        #axs[2].plot([0,0], [np.amin(vary), np.amax(vary)], c = 'w')
    #yl = ['']
    #for i in range(3):
        #else:
        #axs[i].set_xscale('linear')
        #axs[i].set_ylabel(namex)
        #axs[i].grid(True)

def plotNDim2(PS):
    ndd = PS.nondimDict
    namex, namey = ndd['pars']
    if PS.gp['Ori']:
        namex, namey = namex+'Ori', namey+'Ori'
    cmap = 'Blues'
    fig, axs = plt.subplots(3,1, sharex = True)
    Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xs)), np.squeeze(PS.Reg)
    nps = PS.nps
    if 'Fw' in namex:
        varx = symlog10(getattr(PS, namex).reshape(PS.nps))
    else:
        varx = np.log10(getattr(PS, namex).reshape(PS.nps))
    if 'Fw' in namey:
        vary = symlog10(getattr(PS, namey).reshape(PS.nps))
    else:
        vary = np.log10(getattr(PS, namey).reshape(PS.nps))
    
    Regcolor = np.swapaxes(np.reshape(Reg, (*nps, 3)), 0, 1)
    plt.suptitle(PS.name)
    #plt.tight_layout()
    axs[0].imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'lower', aspect = 'auto')
    axs[0].contourf(varx, vary, PS.mask[:,5].reshape(PS.nps), 1, hatches = [" ", "//"], alpha = 0)
    axs[0].contourf(varx, vary, PS.mask[:,6].reshape(PS.nps), 1, hatches = [" ", "."], alpha = 0)
    axs[0].title.set_text('Regime')
    #if 'Fw' in namex:
        #axs[0].plot([0,0], np.array([np.amin(vary), np.amax(vary)]), c = 'w')
    #plt.xticks([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5], ['$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$', '$10$', '$10^2$', '$10^3$', '$10^4$', '$10^5$'])
    #plt.yticks([-3,-2,-1,0], ['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])

    hl = [np.min([np.amin(Phi_0),0]), np.max([np.amin(Phi_0),0]), np.min([np.amax(Phi_0),1]), np.max([np.amax(Phi_0),1])]
    hl.sort()
    minz, maxz = np.min(Phi_0), np.max(Phi_0)
    cfig = axs[1].contourf(varx, vary, Phi_0.reshape(PS.nps), 20, cmap = cmap, corner_mask = True, vmin = minz, vmax = maxz) 
    axs[1].contourf(varx, vary,  PS.mask[:,5].reshape(PS.nps),1, hatches = [" ", "//"], alpha = 0)
    axs[1].contourf(varx, vary, PS.mask[:,6].reshape(PS.nps), 1, hatches = [" ", "."], alpha = 0)
    axs[1].title.set_text(r'Stratification $\Phi_0$')
    plt.colorbar(cfig, ax = axs[1])
    #if 'Fw' in namex:
        #axs[1].plot([0,0], np.array([np.amin(vary), np.amax(vary)]), c = 'w')
        
    minz, maxz = np.amin(Ls), np.amax(Ls)
    cfig2 = axs[2].contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Ls.reshape(PS.nps), 50, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True)
    axs[2].contourf(varx, vary, PS.mask[:,5].reshape(PS.nps), 1, hatches = [" ", "//"], alpha = 0)
    axs[2].contourf(varx, vary,  PS.mask[:,6].reshape(PS.nps),1, hatches = [" ", "."], alpha = 0)
    axs[2].title.set_text(r'Salt intrusion $\Lambda_s$')
    axs[2].set_xlabel(ndd['pars'][0])
    plt.colorbar(cfig2, ax = axs[2])
        
    for i in range(3):
        axs[i].set_ylabel(ndd['pars'][1])
        if 'Fw' in namex:
            axs[i].plot([0,0], [np.amin(vary), np.amax(vary)], c = 'w') 


def plotDim1(PS, dimDict):
    col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    labT = ['G-G', 'G-R', 'G-W', 'R-R', 'R-W', 'W-W', 'DI', '|FL|']
    namex = dimDict['pars'][0]
    #tm = np.transpose(np.array([PS.mask[:,5], PS.mask[:,5]], dtype = bool))
    # Prepare background color for plots
    
    #Reg = ma.masked_array(Reg, mask = tm)
    
    LsT = PS.LsT
    
    fig, axs = plt.subplots(4,1, sharex = True)
    #fig.suptitle('Se)
    plt.tight_layout()

    Phi_0, Ls, Reg = np.squeeze(PS.Phi_0), np.abs(np.squeeze(PS.Xs)), np.squeeze(PS.Reg)
    LDim = dimDict['K_H']/dimDict['c']
    LsDim = Ls*LDim

    if namex == 'tau_w':
        varx = symlog10(dimDict[namex])
    else:
        varx = np.log10(dimDict[namex])

    
    Regcolor = np.array([Reg, Reg])
    extentp = (np.amin(varx), np.amax(varx), np.amin(Phi_0), np.amax(Phi_0))
    Ll, Lu = np.amin(Ls)/1.25, 1.25*np.amax(Ls)
    extentl = (np.amin(varx), np.amax(varx), Ll, Lu)
    extentld = (np.amin(varx), np.amax(varx), Ll*LDim, Lu*LDim)
    #x, y = x.ravel(), y.ravel()
    
    LD = np.ma.masked_array(LsT[:,6])
    LD = np.ma.masked_outside(LD, Ll, Lu)
    
    LGG = np.ma.masked_array(LsT[:,0])
    LGG = np.ma.masked_outside(LGG, Ll, Lu)
    
    LWW = np.ma.masked_array(LsT[:,5])
    LWW = np.ma.masked_outside(LWW, Ll, Lu)
    
    axs[0].plot(varx, Phi_0, lw = 2, c = 'k')
    axs[0].title.set_text(r'Mouth stratification $\Phi_0$')
    f1 = axs[0].imshow(Regcolor, extent = extentp, origin = 'upper', aspect = 'auto')
    if 'tau_w' in namex:
        axs[0].plot([0,0], np.array([np.amin(Phi_0), np.amax(Phi_0)]), c = 'w')
    #cb = plt.colorbar(f1, ax = axs[0], ticks = [1 ,2, 3, 4])
    
    
    axs[1].plot(varx, Ls, lw = 2, c = 'k', label = r'$L_s$')
    axs[1].title.set_text(r'Salt intrusion $\Lambda_s$ (NonDim)')
    f2 = axs[1].imshow(Regcolor, extent = extentl, origin = 'upper', aspect = 'auto')
    axs[1].plot(varx, LD, ls = '-', lw = 1, c = 'w', label = 'Dispersive') #Dispersive Regime
    axs[1].plot(varx, LGG, ls = '--', lw = 1, c = 'w', label = 'Chatwin') #Chatwin Regime
    axs[1].plot(varx, LWW, ls = '-.', lw = 1, c = 'w', label = 'Wind-driven') #WW Regime
    axs[1].set_yscale('log')
    if 'tau_w' in namex:
        axs[1].plot([0,0], np.array([np.amin(Ls), np.amax(Ls)]), c = 'w')
    axs[1].legend()

    axs[2].plot(varx, LsDim, lw = 2, c = 'k', label = r'$L_s$')
    axs[2].title.set_text(r'Salt intrusion $L_s$ (Dim)')
    axs[2].imshow(Regcolor, extent = extentld, origin = 'upper', aspect = 'auto')
    axs[2].plot(varx, LD*LDim, ls = '-', lw = 1, c = 'w', label = 'Dispersive') #Dispersive Regime
    axs[2].plot(varx, LGG*LDim, ls = '--', lw = 1, c = 'w', label = 'Chatwin') #Chatwin Regime
    axs[2].plot(varx, LWW*LDim, ls = '-.', lw = 1, c = 'w', label = 'Wind-driven') #WW Regime
    axs[2].set_yscale('log')
    if 'tau_w' in namex:
        axs[2].plot([0,0], np.array([np.amin(LsDim), np.amax(LsDim)]), c = 'w')
    axs[2].legend()
    # Here, insert theoretical prediction.
    
    T = PS.T
    for ind in range(T.shape[1]-1): axs[3].plot(varx, np.squeeze(T[:,ind]), color=col[ind], label = labT[ind])
    ind = T.shape[1]-1
    axs[3].plot(varx, np.abs(np.squeeze(T[:,ind])), color = col[ind], label = labT[ind])
    axs[3].legend()
    axs[3].title.set_text('Transports')
    axs[3].set_xlabel(dimDict['pars'][0])
    
    #for i in range(4):
        #if namex == 'tau_w':
            #axs[i].set_xscale('linear')
        #else:
            #axs[i].set_xscale('linear')
        #axs[i].set_xlabel(namex)
        #axs[i].grid(True)

    
def plotDim2(PS, dimDict):
    namex, namey = dimDict['pars']
    cmap = 'Blues'
    fig, axs = plt.subplots(4,1, sharex = True)
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
    #vary = np.log10(dimDict[namey].reshape(PS.nps))
    plt.tight_layout()
    
    axs[0].imshow(Regcolor, extent = (np.amin(varx), np.amax(varx), np.amin(vary), np.amax(vary)), origin = 'upper', aspect = 'auto')
    axs[0].title.set_text('Regime')
    axs[0].contourf(varx, vary, PS.mask[:,5].reshape(PS.nps), 1, hatches = [" ", "//"], alpha = 0)
    axs[0].contourf(varx, vary, PS.mask[:,6].reshape(PS.nps), 1, hatches = [" ", "."], alpha = 0) 


    hl = [np.min([np.amin(Phi_0),0]), np.max([np.amin(Phi_0),0]), np.min([np.amax(Phi_0),1]), np.max([np.amax(Phi_0),1])]
    hl.sort()
    minz, maxz = np.min(Phi_0), np.max(Phi_0)
    cfig = axs[1].contourf(varx, vary, Phi_0.reshape(PS.nps), 20, cmap = cmap, corner_mask = True, vmin = minz, vmax = maxz)
    axs[1].contourf(varx, vary, PS.mask[:,5].reshape(PS.nps), 1, hatches = [" ", "//"], alpha = 0)
    axs[1].contourf(varx, vary, PS.mask[:,6].reshape(PS.nps), 1, hatches = [" ", "."], alpha = 0)
    #axs[1].contour(cfig,  levels = np.linspace(minz, maxz), colors = 'k', linewidths = 0.5)
    #axs[1].contour(cfig, levels = [0,1], colors='k', linewidths = 1)
    #axs[1].contourf(varx, vary, Phi_0, levels = hl, vmin = minz, vmax = maxz, hatches = [".", "", "//"], alpha = 0)
    #axs[1].scatter(SM.Ra, SM.Fr, color = 'k', marker = 'o')    #axs[1,2].clabel(cfigc, cfigc.levels, inline = True, fontsize = 10)       
    axs[1].title.set_text(r'Stratification $\Phi_0$')
    plt.colorbar(cfig, ax = axs[1])

    minz, maxz = np.amin(Ls), np.amax(Ls)
    #print(np.amin(Ls))
    #cfig2 = axs[2,2].contourf(Ra, Fr, Ls, 20, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True, vmin = minz, vmax = maxz)
    cfig2 = axs[2].contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), Ls.reshape(PS.nps), 50, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True)
    axs[2].contourf(varx, vary, PS.mask[:,5].reshape(PS.nps), 1, hatches = [" ", "//"], alpha = 0)
    axs[2].contourf(varx, vary, PS.mask[:,6].reshape(PS.nps), 1, hatches = [" ", "."], alpha = 0)   
    axs[2].title.set_text(r'Salt intrusion $\Lambda_s$')
    plt.colorbar(cfig2, ax = axs[2])
    
    LDim = dimDict['K_H']/dimDict['c']*np.ones_like(Ls)
    LsDim = Ls*LDim
    
    minz, maxz = np.amin(LsDim), np.amax(LsDim)
    #print(np.amin(Ls))
    #cfig2 = axs[2,2].contourf(Ra, Fr, Ls, 20, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True, vmin = minz, vmax = maxz)
    cfig3 = axs[3].contourf(varx.reshape(PS.nps), vary.reshape(PS.nps), LsDim.reshape(PS.nps), 50, locator=ticker.LogLocator(), cmap = cmap, corner_mask = True)
    axs[3].contourf(varx, vary, PS.mask[:,5].reshape(PS.nps), 1, hatches = [" ", "//"], alpha = 0)
    axs[3].contourf(varx, vary, PS.mask[:,6].reshape(PS.nps), 1, hatches = [" ", "."], alpha = 0)     
    axs[3].title.set_text(r'Salt intrusion $L_s$')
    axs[3].set_xlabel(dimDict['pars'][0])
    plt.colorbar(cfig3, ax = axs[3])
    
    for i in range(4):
        if 'tau_w' in namey:
            #axs[i].set_xscale('symlog')
            axs[i].plot([0,0], [np.amin(vary), np.amax(vary)], c = 'w') 
        #else:
            #axs[i].set_xlabel(namex)
            
        if 'tau_w' in namey:
            axs[i].set_ylabel(r'$\tau_w$')
        else:
            axs[i].set_ylabel(dimDict['pars'][1]) 


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
    
def plotSModel(SM):
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
    
def addC(R):
    C = np.zeros(12)
    if R != 'Inf':
        C[0] = (19*R**2+285*R+1116)/(1451520*R**2 + 8709120*R + 13063680) #-P2P5
        C[1] = (19*R**2+153*R)/(20160*R**2 + 120960*R + 181440) #-P1P5 = -P2P4
        C[2] = (7*R**2+91*R+306)/(40320*R**2 + 241920*R + 362880) #-P3P5 = -P2P6
        C[3] = (2*R**2)/(105*R**2 + 630*R + 945) #-P1P4
        C[4] = (5*R**2+31*R)/(840*R**2 + 5040*R + 7560) #-P1P6 = -P3P4
        C[5] = (R**2+11*R+32)/(1680*R**2 + 10080*R + 15120) #-P3P6

        C[6] = -7/120*R/(R+3) #P4(0)
        C[7] = -1/2880*(5*R+36)/(R+3) #P5(0)
        C[8] = -1/240*(3*R+16)/(R+3) #P6(0)
        C[9] = R/(R+3)*(-1/8 + 1/4 - 7/120) #P4(-1)
        C[10] = 1/120 - (R+4)/(R+3)/64+(R+6)/(R+3)/96 + C[7]  #P5(-1)
        C[11] = (R+2)/(R+3)/16 - 1/6 + (R+4)/(R+3)/8+C[8] #P6(-1)
    
    else:
        C[0] = 19/1451520 #P2P5
        C[1] = 2*19/40320 #P1P5 = P2P4
        C[2] = 2*1/11520 #P3P5 = P2P6
        C[3] = 2/105 #P1P4
        C[4] = 2*1/336 #P1P6 = P3P4
        C[5] = 1/1680 #P3P6

        C[6] = -7/120 #P4(0)
        C[7] = -1/576 #P5(0)
        C[8] = -1/80 #P6(0)
        C[9] = 1/15 #P4(-1)
        C[10] = 1/720 #P5(-1)
        C[11] = 1/120 #P6(-1) 
    return C

def formFunctions(sigmap, R):
    if R != 'Inf':    
        R3 = 1/(R+3)
        P1 = R*R3*(1/2-3/2*sigmap**2)
        P2 = 1/48*(R+6)*R3 - 3/16*R3*(R+4)*sigmap**2 - 1/6*sigmap**3
        P3 = (R+4)*R3/4 + sigmap + 3*(R+2)*R3/4*sigmap**2
        P4 = R*R3*(-7/120 + sigmap**2/4 - sigmap**4/8)
        P5 = -(5*R+36)*R3/2880 + 1/96*R3*(R+6)*sigmap**2 - 1/64*(R+4)*R3*sigmap**4 - 1/120*sigmap**5
        P6 = -1/240*R3*(3*R+16) + 1/8*R3*(R+4)*sigmap**2 + sigmap**3/6 + (R+2)*R3*sigmap**4/16
        P7 = -1/48*(5*R+18)*R3 - 1/16*(R+4)*R3*sigmap**3 - 1/24*sigmap**4
        
    else:
        P1 = -3/2*sigmap**2 + 1/2
        P2 = -sigmap**3/6 - 3/16*sigmap**2 + 1/48
        P3 = 3/4*sigmap**2 + sigmap + 1/4
        P4 = -1/8*sigmap**4 + sigmap**2/4 - 7/120
        P5 = -sigmap**5/120 - sigmap**4/64 + sigmap**2/96 - 1/576
        P6 = sigmap**4/16 + sigmap**3/6 + sigmap**2/8 - 1/80
        P7 = -5/48 - 1/24*sigmap**4 - 1/16*sigmap**3
    
    return P1, P2, P3, P4, P5, P6, P7

#integral functions for computing transport terms

    
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