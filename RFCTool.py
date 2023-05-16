import numpy as np

GeV = 1.78266192e-27
amu = 931.494e-3 * GeV
e   = 1.602e-19
pi  = 3.1428
kB  = 1.380649e-23
N0  = 2.69e25


class RFCSettings:
    def __init__(self, p=0.25*1e-3,N=2,Epush=1000,q=1,m=133,T=293,Vpp=400,Omega=8e6 *2*pi,mu=0.0018,yloss=-1 ):
        self.p     = p             # Carpet pitch (electrode center to electrode center) in m
        self.N     = N             # N phases
        self.Epush = Epush         # Push field strength in V/m
        self.q     = q             # ion charge in e
        self.m     = m             # ion mass in amu
        self.T     = T             # Buffer gas temperature in K
        self.Vpp   = Vpp           # Peak-to-peak voltage in V
        self.Omega = Omega         # RF Frequency in rad s^-1
        self.mu    = mu            # Mobility in m^2 / Vs
        self.yloss = yloss         # Loss surface (-1 automtically fixes to paper default, 0.25 pitches)


class RFC:

    def __init__(self,S=RFCSettings()):
        self.Update(S)

    def Update(self,S):
        self.p     = S.p                            # Carpet pitch
        self.N     = S.N                            # N phases
        self.Epush = S.Epush                        # Push field (V/m)
        self.q     = S.q*e                          # Ion charge
        self.T     = S.T                            # Temperature (K)
        self.Vpp   = S.Vpp                          # Peak-to-peak volage (V)
        self.mu    = S.mu                           # Mobility (SI)
        self.m     = S.m *amu                       # Ion mass (amu)
        self.Omega = S.Omega                        # Frequency (rad^-1)
        self.D     = self.q/(self.mu*self.m)
        self.eta   = np.arctan(-self.D/S.Omega)
        if(S.yloss<0):
            self.yloss = 0.25*self.p+1e-5
        else:
            self.yloss=S.yloss

        #Some internal calculations of the prob distribution
        self.spacing=0.001
        self.ys=np.arange(self.yloss,20*self.p,self.spacing*self.p)
        self.EffPots=self.CalcV(self.ys)
        self.Distn=np.exp(-(self.EffPots-min(self.EffPots))/(kB*self.T))
        self.PartitionFn=sum(self.Distn)*self.spacing*self.p


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
        if(norm==0):
            NormFactor=1./self.PartitionFn
        elif(norm==1):
            NormFactor=1./max(Distn)
        return np.exp(-(self.CalcV(y)-min(self.EffPots))/(kB*self.T)) *NormFactor


    def IntegralDistLoss(self):
        return(1.-sum(self.Distn*(self.ys>self.yloss))/sum(self.Distn))

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
