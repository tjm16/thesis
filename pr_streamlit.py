import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def dNdt(t,N,N0,V,n,F,k,P0,Pn,m):
    """
    Carbon mass balance equation
    dN/dt = V - Ws - F(t)

    Inputs:
        t: time since LIP emplacement [Myr]
        N: carbon in surficial system [examol]
        N0: initial carbon in surficial system [examol]
        V: volcanic degassing flux [examol/Myr]
        n: strength of global silicate weathering feedback [unitless]
        F: perturbation forcing function

    Outputs:
        dN/dt surficial carbon flux [examol/Myr]
    """
    y = (N/N0)**2
    W = (V-k)*y**n # change V
    #npp0 = r*c*nu*P0
    #nppt = r*c*nu*Pn
    Bg = k*(Pn/P0)**m 
    return V - W - F(t,y) - Bg, Bg, W

def LIP_F(t,y,A0,KCO2,tau,n_LIP,burst=0):
    """
    LIP forcing function

    Inputs:
        t: time since emplacement at model run [Myr]
        y: normalized pCO2 [unitless]
        A0: initial area [Mkm2]
        KCO2: CO2 consumption rate [examol/Mkm2/Myr]
        tau: decay timescale [Myr]
        n_LIP: LIP silicate weathering feedback strength [unitless]
        burst: initial instantaneous volcanic degassing at emplacement [examol]

    Outputs:
        F: carbon flux [examol/Myr]
    """
    A = A0*np.exp(-t/tau) # effective area
    F = KCO2*A*y**n_LIP
    
    # initial volcanic outgassing from LIP formation
        #  set to zero as default here for simplicity
    if t==0:
        F -= burst # note that this sign gets flipped back in the dNdt function

    return F

def dPdt(A,h,rho,chi_m,m_P,tau,t):
    return (A * h * rho * (chi_m / m_P) * np.exp(-t/tau)) / (10**12)
    
def run_model(N0,V,n,
              A0,KCO2,tau,n_LIP,
              Ar,h,rho,chi_m,m_P,
              P0,k,m,
              t_step=0.01, # time step [Myr]
              max_t=30): # how long to run model [Myr]
    """
    Integrates the model using forward euler

    Inputs:
        N0,V,n,A0,KCO2,tau,n_LIP as above
        t_step: time step for integration [Myr]
        max_t: time after emplacement at which to stop model [Myr]

    Outputs:
        t: times at which equation is evaluated (array of [Myr])
        N: carbon in surficial system at each time (array of [examol])
    """
    # initial conditions
    t = [0]
    N = [N0]
    # add phosphorus
    P = [P0]
    # simplified versions of relevant functions that only take in non-constant parameters
    F = lambda t,y: LIP_F(t,y,A0=A0,KCO2=KCO2,tau=tau,n_LIP=n_LIP)
    CF0, OG0, WF0 = dNdt(t[-1],N[-1],N0=N0,V=V,n=n,F=F,k=k,P0=P[0],Pn=P[-1],m=m)
    Wg = [WF0]
    Bg = [OG0]
    CFvec = [CF0]
    # forward Euler
    while t[-1] < max_t:
        # find new phosophorus flux 
        Pflux = dPdt(Ar,h,rho,chi_m,m_P,tau,t[-1])
        P_next = P[-1]+t_step*Pflux
        P.append(P_next)
        CF, OG, WF = dNdt(t[-1],N[-1],N0=N0,V=V,n=n,F=F,k=k,P0=P[0],Pn=P[-1],m=m)
        N_next = N[-1]+t_step*CF
        N.append(N_next)
        t.append(t[-1]+t_step)
        Bg.append(OG)
        Wg.append(WF)
        CFvec.append(CF)
    

    return np.array(t),np.array(N),np.array(P),np.array(Bg),np.array(Wg),np.array(CFvec)

# Set values (these are just some defaults for playing with the model - we'll talk more about where these come from)
N0 = 2.8 # initial surficial carbon [examol] - this is set around present-day 
V = 7.5 # volcanic degassing flux [examol/Myr]
n = 0.5 # global silicate weathering feedback strength (0-1 but usually 0.2-0.7)
A0 = 2.62 # LIP initial area [Mkm2]
KCO2 = 1 # LIP CO2 consumption rate [examol/Myr/Mkm2]
tau = 5 # LIP decay time scale [Myr]
n_LIP = 0.5 # LIP-specific silicate weathering feedback strength

# interface for app
st.title("Thesis Progress Repoort Results")

# new variables

P0 = st.slider("P_0 [examol]", min_value=0.0000, max_value=0.0050, value=0.0020)
Ar = st.slider("A [m^2]", min_value=2.0*10**6, max_value=5.0*10**6, value=4.0*10**6)
h = st.slider("h [m/yr]", min_value=0.0010, max_value=0.0030, value=0.0010)
k = st.slider("k [Emol/Myr]", min_value=1.00, max_value=4.00, value=3.75)
m = st.slider("m [dimensionless]", min_value=1.00, max_value=3.00, value=2.00)

# phosphorus flux
rho = 3000 # density of LIP [kg/m3]
chi_m = 0.001 # percentage P of basalt 
m_P = 0.124 # molar mass of basalt [kg/mol]

# run model
t,N,P,Bg,Wg,tCflux = run_model(N0=N0,V=V,n=n,A0=A0,KCO2=KCO2,tau=tau,n_LIP=n_LIP,
                Ar=Ar,h=h,rho=rho,chi_m=chi_m,m_P=m_P,P0=P0,k=k,m=m)

# plot results
fig,ax = plt.subplots()
ax.set_xlabel('time [Myr]')
ax.set_ylabel('N [examol]')
ax.set_title('Initial Model Results')
ax.plot(t,N,label="Surficial Carbon")


fig,ax = plt.subplots()
ax.set_xlabel('time [Myr]')
ax.set_ylabel('per mille')
ax.set_title('delta 13 C "proxy"')
ax.plot(t,Bg/Wg,label="delta 13 C proxy")
#plt.show()

fig,ax = plt.subplots()
ax.set_xlabel('time [Myr]')
ax.set_ylabel('C Flux [Emol/Myr]')
ax.set_title('Total Carbon ROC')
ax.plot(t,tCflux,label="total carbon flux")
#plt.show()

fig, axs = plt.subplots(4, 1, figsize=(8, 4 * 4))

axs[1].plot(t,N)
axs[1].set_title("Initial Model Results")
axs[1].set_xlabel("time [Myr]")
axs[1].set_ylabel("N [examol]")

axs[2].plot(t,tCflux)
axs[2].set_title("Total Carbon Flux")
axs[2].set_xlabel("time [Myr]")
axs[2].set_ylabel("dN/dt [examol/Myr]")

axs[3].plot(t,Bg/Wg)
axs[3].set_title("delta 13 C 'proxy'")
axs[3].set_xlabel("time [Myr]")
axs[3].set_ylabel("Organic/Inorganic BR")

axs[4].plot(t,P)
axs[4].set_title("Oceanic Phosphorus")
axs[4].set_xlabel("time [Myr]")
axs[4].set_ylabel("P [examol]")

plt.tight_layout()
st.pyplot(fig)
