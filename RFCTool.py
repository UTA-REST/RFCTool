import numpy as np

GeV = 1.78266192e-27
amu = 931.494e-3 * GeV
e   = 1.602e-19
pi  = 3.1428
kB  = 1.380649e-23
N0  = 2.69e25


class RFC:

    def __init__(self,p=0.25*1e-3,N=2,Epush=10*1e2,q=1,m=133,T=293,Vpp=400,Omega=8*2*3.1428*2*1e6,mu=0.0018,yloss=-1):
        self.p     = p                            # Carpet pitch
        self.N     = N                            # N phases
        self.Epush = Epush                        # Push field (V/m)
        self.q     = q*e                          # Ion charge
        self.T     = T                            # Temperature (K)
        self.Vpp   = Vpp                          # Peak-to-peak volage (V)
        self.mu    = mu                           # Mobility (SI)
        self.m     = m *amu                       # Ion mass (amu)
        self.Omega = Omega                        # Frequency (rad^-1)
        self.D     = self.q/(self.mu*self.m)
        self.eta   = np.arctan(-self.D/Omega)
        if(yloss<0):
            self.yloss = 0.25*self.p
        else:
            self.yloss=yloss

    def CalcV(self,y):
        vx=self.q**2/(self.m*(self.D**2+self.Omega**2))*0.5*(2*pi/(self.N*self.p))**2*(self.Vpp/2)**2*np.exp(-4*pi*y/(self.N*self.p)) +self.q*self.Epush*y
        return(vx)


    def CalcY(self):
        Prefactor   = self.N*self.p/(4*pi)
        Denominator = self.Epush * (self.N*self.p)**3 * self.m* (self.Omega**2+self.D**2)
        Numerator   = (2*self.Vpp**2*self.q*(pi)**3)
        y= Prefactor * np.log(Numerator/Denominator)
        return(y)

    def EquilVelocity(self):
        return 0.5*self.mu*self.Epush*self.D/self.Omega

    def DistributionFunction(self,y,norm=0):
        spacing=0.001
        ys=np.arange(self.yloss,20*self.p,spacing*self.p)
        EffPots=self.CalcV(ys)
        Distn=np.exp(-(EffPots-min(EffPots))/(kB*self.T))
        PartitionFn=sum(Distn)*spacing*self.p
        if(norm==0):
            NormFactor=1./PartitionFn
        elif(norm==1):
            NormFactor=1./max(Distn)
        return np.exp(-(self.CalcV(y)-min(EffPots))/(kB*self.T)) *NormFactor


    def IntegralDistLoss(self):
        spacing=0.001
        ys=np.arange(-20*self.p,20*self.p,spacing*self.p)
        EffPots=self.CalcV(ys)
        Distn=np.exp(-(EffPots-min(EffPots))/(kB*self.T))
        return(1.-sum(Distn*(ys>self.yloss))/sum(Distn))

    def VMin(self):
        return np.sqrt(self.m*(self.D**2+self.Omega**2)*self.Epush/(2*self.q)*(self.N*self.p/pi)**3)

    def dVdy(self,y):
        return self.q*Epush-self.q**2/(self.m*(self.D**2+self.Omega**2))*(2*pi/(self.N*self.p))**3*(self.Vpp/2)**2*np.exp(-4*pi/(self.N*self.p)*y)

    def LossFunction(self):
        rho0=self.DistributionFunction(self.yloss)
        v=self.EquilVelocity()
        nvy= np.sqrt(kB*self.T/(2*pi*self.m))
        return 1/v*rho0*nvy

    def IonLossRate(self):
        rho0=self.DistributionFunction(self.yloss)
        nvy= np.sqrt(kB*self.T/(2*pi*self.m))
        return rho0*nvy

    def MicroRadius(self,y):
        return self.q/(self.m*self.Omega*np.sqrt(self.Omega**2+self.D**2))*self.Vpp*pi/(self.N*self.p)*np.exp((-2*pi)/(self.N*self.p)*y)

    def TrapDepth(self):
        yeq=self.CalcY()
        return self.CalcV(yeq)-self.CalcV(0)

    def EquilibriumRadius(self):
        return np.sqrt(self.N*self.p/(2*pi)*self.q*self.Epush/(self.m*self.Omega**2))

    #legacy:

    #def VPrime(self):
    #    return -q/(m*(D**2+Omega**2))*(2*pi/(N*p))**3*(Vpp/2)**2*np.exp(-4*pi/(N*p)*yloss)+Epush

    # def VPrimePrime(N,p,Omega,m,mu,Vpp,Epush,yloss):
    #    return -2*q/(m*(D**2+Omega**2))*(2*pi/(N*p))**4*(Vpp/2)**2*np.exp(-4*pi/(N*p)*yloss)

    #def NegVy(N,p,Omega,m,mu,Vpp,Epush,yloss,kB,T):
    #    D=q/(mu*m)
    #    VPr=VPrime(N,p,Omega,m,mu,Vpp,Epush,yloss)
    #    C=mu*np.sqrt(m/(2*kB*T))*VPr
    #    return -0.5*mu*VPr*sc.special.erfc(-C)-np.sqrt(kB*T/(2*pi*m)*np.exp(-C**2))
