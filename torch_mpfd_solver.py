import torch
import numpy as np
import pandas as pd

### Parameters ###

def makeDictFloat():
    d = {}
    return d

def havercampSetpars():
    havercampPars = makeDictFloat()
    havercampPars['thetaR'] = 0.075
    havercampPars['thetaS'] = 0.287
    havercampPars['alpha'] = 1.611e6
    havercampPars['beta'] = 3.96
    havercampPars['A'] = 1.175e6
    havercampPars['gamma'] = 4.74
    havercampPars['Ks'] = 0.00944
    return havercampPars


def gardnerSetpars():
    gardnerPars = makeDictFloat()
    gardnerPars['thetaR'] = 0.065
    gardnerPars['thetaS'] = 0.45
    gardnerPars['Ks'] = 3.466*10**(-5)
    gardnerPars['lambda'] = 10**(-2)
    gardnerPars['Rmax'] = 1.157*10**(-1)
    gardnerPars['tStop'] = 2.592*10**(5)

    return gardnerPars

### Constitutive relations ###

def havercampCfun(psi, pars):
    x3 = (pars['alpha'] + torch.abs(psi)**pars['beta'])**2.
    x2 = pars['alpha'] / x3
    x1 = pars['beta'] * torch.abs(psi)**(pars['beta'] - 1) * x2
    C = (pars['thetaS'] - pars['thetaR']) * x1
    return C

def havercampKfun(psi, pars):
    x2 = pars['A'] + torch.abs(psi)**pars['gamma']
    x1 = pars['A'] / x2
    K = pars['Ks'] * x1
    return K

def havercampthetafun(psi, pars):
    x3 = pars['alpha'] + torch.abs(psi)**pars['beta']
    x2 = pars['alpha'] / x3
    x1 = (pars['thetaS'] - pars['thetaR']) * x2
    theta = pars['thetaR'] + x1
    return theta

def zeroFun(psi,pars):
    return 0

def gardnerCfun(psi,pars):
    C = pars['lambda']*(pars['thetaS']-pars['thetaR'])*torch.exp(pars['lambda']*psi)
    return C


def gardnerKfun(psi,pars):
    K = pars['Ks']*torch.exp(pars['lambda']*psi)

    return K


def gardnerthetafun(psi,pars):
    theta = (pars['thetaS']-pars['thetaR'])*torch.exp(pars['lambda']*psi) + pars['thetaR']
    return theta

def gardnerhfun(theta,pars):
    psi = (np.log(theta-pars['thetaR']) - np.log(pars['thetaS']-pars['thetaR']))/pars['lambda']
    return psi


def gardnersinkfun(psi,pars):
    h1 = -0.1
    h2 = -0.25
    h3 = -5
    h4 = -160
    S_h = torch.zeros_like(psi)
    
    condition1 = (psi <= 0) | (psi >= h4)
    condition2 = (h2 < psi) & (psi <= h1)
    condition3 = (h3 <= psi) & (psi <= h2)
    condition4 = (h4 < psi) & (psi <= h3)
    
    S_h = torch.where(condition2, pars['Rmax']* (psi - h1) / (h2 - h1), S_h)
    S_h = torch.where(condition3, pars['Rmax'], S_h)
    S_h = torch.where(condition4, pars['Rmax'] * (psi - h4) / (h3 - h4), S_h)
    return S_h


def gardnertopboundaryfun(t,pars):
    theta = (2*pars['tStop']-t)/(2*pars['tStop']) * (0.1*pars['thetaR'] + 0.9*pars['thetaS'])
    return theta

def gardnerinitialprofile(z):
    theta = 0.4115 + z/30 * (0.1035-0.4115)
    return theta



### SOLVER ###

def solverfun(R, C, Kmid, dt, dz, n):
    a = torch.zeros(n, dtype=torch.float32)
    b = torch.zeros(n, dtype=torch.float32)
    c = torch.zeros(n, dtype=torch.float32)
    y = torch.zeros(n, dtype=torch.float32)

    a = Kmid[:-1] / dz
    b = -(Kmid[:-1] + Kmid[1:]) / dz - C * dz / dt
    c = Kmid[1:] / dz
    A = torch.diag(a[1:], -1) + torch.diag(b, 0) + torch.diag(c[:-1], 1)
    
    y[:] = R[:]

    dell = torch.linalg.solve(A, y)

    return dell

def Rfun(psiiter, psiin, psiT, psiB, C, Kmid, dtheta, dt, dz, n,sink,pars):
    
    psigrid = torch.cat((psiB, psiiter, psiT))
    x1 = dtheta / dt * dz
    x2 = -(Kmid[1:] - Kmid[:-1])
    x3 = -Kmid[1:] * (psigrid[2:] - psigrid[1:-1]) / dz
    x4 = Kmid[:-1] * (psigrid[1:-1] - psigrid[:-2]) / dz

    R = x1 + x2 + x3 + x4 + sink(psiin,pars)

    return R


def neumannIterFun(psiin, pars, psiT, psiB, dt, dz, n,Cfun,Kfun,thetafun,sink):
    tolerance = 1e-10
    maxcount = 1000
    Rmax = 1.

    psiiter = psiin.clone()
    psiout = psiin.clone()

    count = 0
    while count <= 1 or (Rmax >= tolerance and count <= maxcount):
        C = Cfun(psiiter, pars)

        K = Kfun(torch.cat((psiiter[0]*dz,psiiter,psiT)),pars)

        Kmid = (K[1:] + K[:-1]) / 2.
        dtheta = thetafun(psiiter, pars) - thetafun(psiin, pars)

        R = Rfun(psiiter, psiin, psiT, psiiter[0]*dz, C, Kmid, dtheta, dt, dz, n,sink,pars)

        dell = solverfun(R, C, Kmid, dt, dz, n)
        psiout = psiiter + dell
        psiiter = psiout
        #print(psiiter)
        Rmax = torch.abs(torch.max(R))
        count += 1

    return psiout

def dirichletIterFun(psiin, pars, psiT, psiB, dt, dz, n,Cfun,Kfun,thetafun,sink):
    tolerance = 1e-10
    maxcount = 1000
    Rmax = 1.

    psiiter = psiin.clone()
    psiout = psiin.clone()

    count = 0
    while count <= 1 or (Rmax >= tolerance and count <= maxcount):
        C = Cfun(psiiter, pars)


        K = Kfun(torch.cat((psiB, psiiter, psiT)), pars)


        Kmid = (K[1:] + K[:-1]) / 2.
        dtheta = thetafun(psiiter, pars) - thetafun(psiin, pars)


        R = Rfun(psiiter, psiin, psiT, psiB, C, Kmid, dtheta, dt, dz, n,sink,pars)


        dell = solverfun(R, C, Kmid, dt, dz, n)
        psiout = psiiter + dell
        psiiter = psiout
        #print(psiiter)
        Rmax = torch.abs(torch.max(R))
        count += 1

    return psiout


def massBalance (output,Kfun,thetafun,dz,dts,pars):
    psi = torch.stack(output)

    dS = torch.sum(thetafun(psi[-1],pars)-thetafun(psi[0],pars))*dz
    
    Kin = (Kfun(psi[:,0],pars) + Kfun(psi[:,1],pars))/2
    Qin = -Kin * ((psi[:, 1] - psi[:,0]) / dz + 1.)
    Qin[0] = 0
    QinSum = torch.dot(Qin,torch.hstack([dts[0],dts]))

    Kout = (Kfun(psi[:,-1],pars) + Kfun(psi[:,-2],pars))/2
    Qout = -Kout * ((psi[:,-1] - psi[:, -2]) / dz + 1.)
    Qout[0] = 0
    QoutSum = torch.dot(Qout,torch.hstack([dts[0],dts]))

    err = (dS)/(QinSum-QoutSum)

    return err


def dirichletOneStepModelRun(dt,dz,n,psi,psiB,psiT,pars,Cfun,Kfun,thetafun,sink):

    psiNext = dirichletIterFun(psi,pars,psiT,psiB,dt,dz,n,Cfun,Kfun,thetafun,sink)

    return psiNext


def neumannOneStepModelRun(dt,dz,n,psi,psiB,psiT,pars,Cfun,Kfun,thetafun,sink):

    psiNext = neumannIterFun(psi,pars,psiT,psiB,dt,dz,n,Cfun,Kfun,thetafun,sink)

    return psiNext

def fullModelRun(t,dts,dz,n,nt,psi,psiB,psiT,pars,Cfun,Kfun,thetafun,flag,sink):
    psiList = []
    psiList +=[psi]

    if flag==0:
        for j in range(1,nt):
            psiList += [dirichletOneStepModelRun(dts[j-1],dz,n,psiList[j-1],psiB[j-1],psiT[j-1],pars,Cfun,Kfun,thetafun,sink)]
        

    
    elif flag==1:
        for j in range(1,nt):
            psiList += [neumannOneStepModelRun(dts[j-1],dz,n,psiList[j-1],dz*psiB[j-1],psiT,pars,Cfun,Kfun,thetafun,sink)]



    else:
        raise ValueError("flag should be either 0 or 1, corresponding to Dirichlet or Neumann boundary condition at the bottom node.")
    
    return psiList



def outputWrapper(psiList,flag,psiB,psiT):
    if flag==0:
        output = [torch.hstack([psiB[i],psiList[i],psiT[i]]) for i in range(len(psiList))]
    
    elif flag==1:
        output = [torch.hstack([psiList[i][0],psiList[i],psiT[i]]) for i in range(len(psiList))]

    return output




def setup(dt, tN, zN, psiInitial,setpars):
    #works only for constant dirichlet boundary conditions, as is.
    pars = setpars()

    dz = zN / (len(psiInitial) - 1)
    #dz = torch.tensor([zN / (len(psiInitial) - 1)],dtype=torch.float32)
    z = torch.arange(dz, zN, dz)
    dz = torch.tensor([dz],dtype=torch.float32)
    n = len(z)

    t = torch.arange(0, tN, dt)
    t = torch.hstack([t,torch.tensor(tN)])
    nt = len(t)

    dts = torch.stack([t[i+1] - t[i] for i in range(len(t)-1)])
    
    #psi_list = []
    #psi = torch.zeros((nt, n), dtype=torch.float32)
    #psi[0, :] = torch.tensor(psiInitial[1:-1], dtype=torch.float32)
    psi = torch.tensor(psiInitial[1:-1], dtype=torch.float32)
    psi.requires_grad = True

    #psiB = torch.tensor([psiInitial[0]], dtype=torch.float32)
    psiB = [torch.tensor([psiInitial[0]], dtype=torch.float32) for _ in t]
    psiT = [torch.tensor([psiInitial[-1]], dtype=torch.float32) for _ in t]
    
   

    #psi_list.append(psi)

    return z, t, dts, dz, n, nt, zN, psi, psiB, psiT, pars


