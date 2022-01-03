import numpy as np
import matplotlib.colors as co
#from numpy.linalg.linalg import qr
#from numpy.ma.extras import masked_all
#import numpy.matlib

def globalParameters(**kwargs):
    gp = dict(R = 2, #R: real or Inf
        ep = 1/30, #Salinity fraction at landward end
        n = [1, 401, 31, 13], #0 parameters to vary, 1 parameter, 2 parameters, 3 parameters.
        SM = [501,41], #Single model run: grid.
        PS = [51,11],
        tolNEG = 0,
        tolUN = 0,
        tolPrit = 10, #not employed in current version of script.
        Sc = 2.2,
        mixAway = False,
        m = 2, #tuning factor for mixing algorithm
        Ori = True, #Plots 'Original' parameters instead of modified ones. (now; only NonDim compatible)
        )
    
    gp['C'] = addC(gp['R'])
    
    for key, value in kwargs.items(): #Change the constant parameters
            gp[key] = value
    return gp

def solveCubic(r):
    '''Computes single real solution based on parameter values'''
    solC = np.roots(r)
    #print(f'Three roots: {solC}')
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
    #tau_w = np.array(invsymlog10(np.linspace(symlog10(-4.3), symlog10(4.3), 101)))

    Q = np.logspace(1,4, n)
    H = np.logspace(0,2.5, n)
    if 'tau_w' in args:
        tau_w  = np.array(invsymlog10(np.linspace(symlog10(-1), symlog10(1), n)))
        if 'tauwLim' in kwargs:
            tau_w  = np.array(invsymlog10(np.linspace(symlog10(kwargs['tauwLim'][0]), symlog10(kwargs['tauwLim'][1]), n)))
        if 'dir' in kwargs:
            if kwargs['dir'] == 'Down':
                tau_w = np.array(invsymlog10(np.linspace(symlog10(0), symlog10(0.3), n)))
            elif kwargs['dir'] == 'Up':
                tau_w = np.array(invsymlog10(np.linspace(symlog10(-0.3), symlog10(0), n)))
    #print(tau_w)
    # Make mixing dependent on H or tau_w, not the other way around!!
    #if ('Ralston' not in kwargs) and ('Sverdrup' not in kwargs):
    K_M = np.logspace(-4,0,n)
    K_H = np.logspace(0,5,n)
    
    if 'Guha' in kwargs:
        if kwargs['Guha']:
            dd['K_M'] = Guha(dd['H'], dd['B'], dd['K_H'])
            print(f"Mixing = {dd['K_M']}")
    if 'Sverdrup' in kwargs: # In this case, make K_M dependent on tau_w following Kullenberg 1976
        if kwargs['Sverdrup']:
            omega = kwargs['Sverdrup']/(2.6e-3*1.225) # Change this coefficient to tune wind.
            dd['K_M'] = Sverdrup(tau_w, dd['K_M'], omega)
    #elif 'Ralston' in kwargs:
    #   K_M, K_H = Ralston(kwargs['Ralston'], H, dd['B'])
    #    if 'Sverdrup' in kwargs:
    #        K_M = Sverdrup(tau_w, K_M, kwargs['Sverdrup'])
    #elif 'Sverdrup' in kwargs:
    #    K_M = Sverdrup(tau_w, dd['K_M'], kwargs['Sverdrup'])
    #    K_H = dd['K_H']
    
    if len(args) == 1:
        varx = eval(args[0])
        dd[args[0]] = varx
        #if ('Ralston' in kwargs) or ('Sverdrup' in kwargs):
        #    dd['K_M'], dd['K_H'] = K_M, K_H
        ndd = dim2nondim(dd)
        ndd['nps'] = n
        
    if len(args) == 2:
        varx, vary = np.meshgrid(eval(args[0]), eval(args[1])) #bad practice but in this case not very harmful
        varx, vary = varx.flatten(), vary.flatten()
        dd[args[0]] = varx
        dd[args[1]] = vary
        #if 'Ralston' in kwargs:
        #    if args[0] == 'H':
        #        dd['K_M'], dd['K_H'] = Ralston(kwargs['Ralston'], varx, dd['B'])
        #    if args[1] == 'H':
        #        dd['K_M'], dd['K_H'] = Ralston(kwargs['Ralston'], vary, dd['B'])
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
    '''Makes nondimensional parameter Dict with default initial values.
    
    makeNDDict(gp, args, kwargs)
    
    Inserting args yields a variation for that argument
    
    Inserting kwargs yields fixed value for that variable, and limitations of the varying variables. '''
    
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
    if 'RaLim' in kwargs:
        Ra = np.logspace(np.log10(kwargs['RaLim'][0]), np.log10(kwargs['RaLim'][1]), n) 
    else:
        Ra = np.logspace(np.log10(25), 5, n) #default
        
    if 'FrLim' in kwargs:
        Fr = np.logspace(np.log10(kwargs['FrLim'][0]), np.log10(kwargs['FrLim'][1]), n) 
    else:
        Fr = np.logspace(-4 , np.log10(2) , n)#default
        
    if 'FwLim' in kwargs:
        Fw = np.array(invsymlog10(np.linspace(symlog10(kwargs['FwLim'][0]), symlog10(kwargs['FwLim'][1]), n)))
    else:
        Fw = np.array(invsymlog10(np.linspace(symlog10(-1), symlog10(8), n)))


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

def Guha(H, B, K_H): # Co
    K_M = 0.00208*K_H*H/B
    return K_M


def Banas(u_T, H, B):
    K_M = 0.028*0.0026*u_T*H
    K_H = 0.035*u_T*B
    return K_M, K_H

def Sverdrup(tau_w, K_M0, beta):
    K_M = K_M0 + beta*np.abs(tau_w)
    return K_M

def computersXs(gp, a, b, c, d, Sb_X0, Fr, Ra, Fw):
    '''
    
    Computes rs and Xs (salt intrusion coordinates)
    
    Input: ODE data
    
    Output: rs, Xs, rs0, Xs0, rs1, Xs1, mask
    
    '''
    ep = gp['ep']
    C = gp['C']
    Sc = gp['Sc']
    #expr, mask = solveCubic(np.array([a*Sb_X0**2, b*Sb_X0, c, -ep*(a*Sb_X0**2 + b*Sb_X0 + c)])) #rt = exp(r)
    
    #depth-averaged salinity intrusion
    expr, mask = solveCubic(np.array([a*Sb_X0**3, b*Sb_X0**2, c*Sb_X0, -ep*d])) #rt = exp(r)
    if (expr < 1.0) and ~mask:
        rs = np.log(expr)
        Xs = (3/2*a*Sb_X0**2*(np.exp(2*rs)-1) + 2*b*Sb_X0*(np.exp(rs)-1) + c*rs)/d
    else:
        mask = True
        return mask, mask, mask,mask, mask, mask,mask
    
    #surface intrusion
    expr0, mask = solveCubic(np.array([a*Sb_X0**3, (b + d*Sc*Ra**2*C[7])*Sb_X0**2, (c+d*Sc*Ra*(Fr*C[6]+Fw*C[8]))*Sb_X0, -ep*d])) #rt = exp(r)
    if (expr0 < 1.0) and ~mask:
        rs0 = np.log(expr0)
        Xs0 = (3/2*a*Sb_X0**2*(np.exp(2*rs0)-1) + 2*b*Sb_X0*(np.exp(rs0)-1) + c*rs0)/d
    else:
        mask = True
        return mask, mask, mask,mask, mask, mask,mask
    
    #bottom intrusion
    expr1, mask = solveCubic(np.array([a*Sb_X0**3, (b + d*Sc*Ra**2*C[10])*Sb_X0**2, (c+d*Sc*Ra*(Fr*C[9]+Fw*C[11]))*Sb_X0, -ep*d])) #rt = exp(r)
    if (expr1 < 1.0) and ~mask:
        rs1 = np.log(expr1)
        Xs1 = (3/2*a*Sb_X0**2*(np.exp(2*rs1)-1) + 2*b*Sb_X0*(np.exp(rs1)-1) + c*rs1)/d
    else:
        mask = True
        return mask, mask, mask,mask, mask, mask,mask
    return rs, Xs, rs0, Xs0, rs1, Xs1, mask

def scaledTransport(gp,L, Sb, Sb_X):
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
    if gp['scaledTransportPlot']:
        TX = [l*p/L[7] for l,p in zip(L, P)] #transports as function of X, scaled by the river lengtscale (Fr)
    else:
        TX = [l*p/L[7]*Sb for l,p in zip(L, P)] #transports as function of X, scaled by the river lengtscale (Fr)
    return TX

def scaledIntegratedTransport(gp, L, a, b, c, d, Sb_X0, rs, Xs):
    ''' Computes integrated (Xs to 0), divided by abs(Xs) and scaled (divided by d*Sigma) transport.
    
    Input: ODE and salt intrusion data
    
    Output: T (numpy array of shape 8)
    
    '''
    ep = gp['ep']
    
    bl, bu = Sb_X0*np.exp(rs), Sb_X0
    if not gp['realTransport']:
        sT = [Int3(bl, bu, a, b, c), #Analytically integrated P = S_X^p / S
            Int2(bl, bu, a, b, c),
            Int2(bl, bu, a, b, c),
            -np.log(ep),
            -np.log(ep),
            -np.log(ep),
            -np.log(ep),
            -Xs
            ]
        T = np.array([l*st/(-Xs*d) for l, st in zip(L, sT)]) #analytical integration, scaled
    else:
        sT = [Sb_X0**3/d*(3/5*a*Sb_X0**2*(1-np.exp(5*rs)) + 1/2*b*Sb_X0*(1-np.exp(4*rs)) + c/3*(1-np.exp(3*rs))), #Analytically integrated P = S_X^p, later scaled by global flushing export instead of in integrand.
            Sb_X0**2/d*(3/4*a*Sb_X0**2*(1-np.exp(4*rs)) + 2/3*b*Sb_X0*(1-np.exp(3*rs)) + c/2*(1-np.exp(2*rs))),
            Sb_X0**2/d*(3/4*a*Sb_X0**2*(1-np.exp(4*rs)) + 2/3*b*Sb_X0*(1-np.exp(3*rs)) + c/2*(1-np.exp(2*rs))),
            Sb_X0/d*(a*Sb_X0**2*(1-np.exp(3*rs)) + b*Sb_X0*(1-np.exp(2*rs)) + c*(1-np.exp(rs))),
            Sb_X0/d*(a*Sb_X0**2*(1-np.exp(3*rs)) + b*Sb_X0*(1-np.exp(2*rs)) + c*(1-np.exp(rs))),
            Sb_X0/d*(a*Sb_X0**2*(1-np.exp(3*rs)) + b*Sb_X0*(1-np.exp(2*rs)) + c*(1-np.exp(rs))),
            Sb_X0/d*(a*Sb_X0**2*(1-np.exp(3*rs)) + b*Sb_X0*(1-np.exp(2*rs)) + c*(1-np.exp(rs))),
            0
            ]
        T = np.array([l*st for l, st in zip(L, sT)]) #analytical integration, scaled
        T[7] = np.sum(T)
        T = T/T[7]
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
    W = -P[6]*Ra**2*np.matlib.repmat(np.transpose(Sb_XX),nsigma,1) # In this, we have assumed the vertical velocity scale K_M / H
    return U, W, Ubar, UR, UG, UW

def computeNU(D2, Exx, Sb_X0, rs):
    maskNU = False
    if D2 >=0:
        #Sb_Xs, mimi = solveCubic(np.array([a, b, c, -d*gp['ep']*Sb_0]))
        Sb_Xs =  Sb_X0*np.exp(rs)
        if (Sb_Xs <= Exx[0] <= Sb_X0) or (Sb_Xs <= Exx[1] <= Sb_X0):
            maskNU = True
    return maskNU



def computePhysicalMasks(gp, Ra, Fr, Fw, Sc, Sb, Sb_X):
    Ssurf, Sbot = computeSurfBottom(gp, Ra, Fr, Fw, Sc, Sb, Sb_X)
    maskNEG = np.any((Ssurf < -gp['tolNEG'])) # 1 if there is negative salt (within tolNEG)
    maskUN = np.any((Sbot - Ssurf < -gp['tolUN'])) # 1 if there exists a point of unstable stratification (within tolUNSTABLE)
    return maskNEG, maskUN

def computePhysicalMasksPS(gp, a, b, c, d, Ra, Fr, Fw, Sc, Sb_X0, Sb_0, rs):
    '''
    Returns maskNEG and maskUNSTABLE (tolerances specified by gp)
    
    Only looks at surface and bottom salinity, not intermediate.
    
    '''
    _,_, Sb, Sb_X, _ = solveODE(gp, a, b, c, d, Sb_X0, rs)
    Ssurf, Sbot = computeSurfBottom(gp, Ra, Fr, Fw, Sc, Sb, Sb_X)
    
    # Central and salt intrusion limit stratification
    nm = int(len(Ssurf)/2)
    Phi_c = Sbot[nm]-Ssurf[nm] # central stratification
    Phi_s = Sbot[0] - Ssurf[0] # salt intrusion stratification
    
    # Surface salt intrusion (done in computersXs)
    
    
    maskNEG = np.any((Ssurf < -gp['tolNEG'])) # 1 if there is negative salt
    #maskUN = (Sbot[-1] - Ssurf[-1] < -gp['tolUN'])
    maxUN = np.array([np.max([0,np.max(Ssurf-Sbot)]), np.argmax(np.abs(Ssurf-Sbot))])
    maskUN = np.any((Sbot - Ssurf < -gp['tolUN'])) # 1 if there exists a point of unstable stratification
    #Phi_s = Ss
    return maskNEG, maskUN, maxUN, Phi_c, Phi_s


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
    Computes scores of Regime 1 (Dispersive), Regime 2 (Chatwin) and 2 (Wind-Driven). Sum of these scores is 1 (by definition)
    '''
    #print(T)
    #s = np.array([T[6], T[0]+T[1]+T[3], abs(T[2]) + abs(T[4]) + abs(T[5])])/np.sum(np.abs(T[0:7]))
    s = np.array([T[0], T[1], abs(T[2]), T[3], abs(T[4]), abs(T[5]), T[6]])/np.sum(np.abs(T[0:7]))
    
    #maxT = np.argmax(np.abs(T[0:7]))
    #alpha = 1.0 - np.sum(s[~np.argmax(s)])
    return s

def computeRegime(T):
    '''
    
    Input: Integrated transport vector
    
    Output: RBG regime vector.
    
    '''
    
    s = computeScore(T)
    #print(s)
    c = np.array([co.to_rgb('tab:green'), co.to_rgb('tab:green'), co.to_rgb('tab:orange'), co.to_rgb('tab:green'), co.to_rgb('tab:orange'), co.to_rgb('tab:orange'), co.to_rgb('tab:blue')]) #Chatwin
    regMat = c#np.array([c1, c2, c3])
    if T[-1] < T[0]:
        r = (1-T[0]+T[-1])*np.matmul(np.transpose(regMat),s) + (T[0]-T[-1])*np.array(co.to_rgb('tab:cyan'))
    else:
        r = np.matmul(np.transpose(regMat),s)
    return r

def computeRegimeOld(T):
    '''
    
    Input: Integrated transport vector
    
    Output: RBG regime vector.
    
    '''
    
    s = computeScore(T)
    c1 = np.array(co.to_rgb('tab:blue')[0:3]) #D
    c3 = np.array(co.to_rgb('tab:orange')[0:3])  #WW
    c2 = np.array(co.to_rgb('tab:green')[0:3]) #Chatwin
    regMat = np.array([c1, c2, c3])
    r = np.matmul(np.transpose(regMat),s)
    return r
    
def computeTheory(gp, L, Sb_0):
    '''Compute theoretical salt intrusion lengths (LsT) (while making use of exact Sb_0 etc (As Dijkstra 2021))
    
    and Sb_X0, Sb_0 and Phi_0 and LsT0 by making use of only regime balances (new)
    
    '''
    ep = gp['ep']
    LsT = np.array([3/2*Sb_0**(2/3)*L[0]**(1/3)/L[7]**(1/3)*(1-ep**2/3),
                2*np.sqrt(L[1])*(np.sqrt(Sb_0) - ep**2*Sb_0**2)/L[7], 
                np.sign(L[2])*2*np.sqrt(np.abs(L[2]))*(np.sqrt(Sb_0) - ep**2*Sb_0**2)/L[7],
                -L[3]/L[7]*np.log(ep),
                -L[4]/L[7]*np.log(ep),
                -L[5]/L[7]*np.log(ep),
                -L[6]/L[7]*np.log(ep),
                np.abs(L[0]*Sb_0/L[2]*(1-ep))] #Balance of GG and up-estuary GW
        )
        
    
        
    return LsT
    
def computeWeM(gp, Ra, Fr, Fw, Sb_X0, Sb_0, Xs):
    R = gp['R']
    We0 = Fw/(Ra*Sb_X0)
    M0 = 4*(-1/8*R/(R+3)*Fr + (-1/30+3/64*(R+4)/(R+3) - 1/96*(R+6)/(R+3))*Ra*Sb_X0 + (-3/16*(R+2)/(R+3) + 1/3 - 1/8*(R+4)/(R+3))*Fw)
    return We0, M0
    
    
#def initMixPar(gp):
    #mixPar = dict()
def symlog10(x,l = 1/(5*np.log(10))):
    return np.sign(x)*np.log10(1.0 + np.abs(x/l))
    
def invsymlog10(y,l = 1/(5*np.log(10))):
    return np.sign(y)*l*(-1.0 + 10**np.abs(y))
    
def addC(R):
    C = np.zeros(12)
    if R != 'Inf':
        C[0] = (19*R**2+285*R+1116)/(1451520*R**2 + 8709120*R + 13063680) #-P2P5
        C[1] = (19*R**2+153*R)/(20160*R**2 + 120960*R + 181440) #-P1P5 = -P2P4 (sum)
        C[2] = (7*R**2+91*R+306)/(40320*R**2 + 241920*R + 362880) #-P3P5 = -P2P6 (sum)
        C[3] = (2*R**2)/(105*R**2 + 630*R + 945) #-P1P4
        C[4] = (5*R**2+31*R)/(840*R**2 + 5040*R + 7560) #-P1P6 = -P3P4 (sum)
        C[5] = (R**2+11*R+32)/(1680*R**2 + 10080*R + 15120) #-P3P6

        C[6] = -7/120*R/(R+3) #P4(0)
        C[7] = -1/2880*(5*R+36)/(R+3) #P5(0)
        C[8] = -1/240*(3*R+16)/(R+3) #P6(0)
        C[9] = R/(R+3)*(-1/8 + 1/4 - 7/120) #P4(-1)
        C[10] = 1/120 - (R+4)/(R+3)/64+(R+6)/(R+3)/96 + C[7]  #P5(-1)
        C[11] = (R+2)/(R+3)/16 - 1/6 + (R+4)/(R+3)/8+C[8] #P6(-1)
        print(f'D4 = {C[9]-C[6]} and D5 = {C[10]-C[7]} and D6 = {C[11]-C[8]} and quot = {(C[9]-C[6])/(C[11]-C[8])}')
        #Sc = 2.2
        #print(Sc*C)
    
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