import numpy as np 
import matplotlib.pyplot as plt
import Model_params as pars
import pandas as pd 
import lmfit 
from scipy.integrate import odeint


## Input parameters 
t = 12 # number of hours to run for
tsteps = 3601 # number of steps
tspan = np.linspace(0,t*60,tsteps)# time mins
Dose = np.array([25,75,100]) # IV doses 
##  Read excel file with data 
LidoDat = pd.read_excel (r'C:\Users\user\Dropbox\Example Code\lidocaine_IVPK_data.xlsx')

## Reshape data to excluse 'NaN' for fit, keeps all as series 
Time_25mg =  LidoDat['Time25mg']
Time_25mg = np.array([x for x in Time_25mg if ~np.isnan(x)])
Conc_25mg =  LidoDat['Conc25mg']*1e-6
Conc_25mg = np.array([x for x in Conc_25mg if ~np.isnan(x)])
Time_75mg =  LidoDat['Time75mg']
Conc_75mg =  LidoDat['Conc75mg']*1e-6
Time_100mg =  LidoDat['Time100mg']
Time_100mg = np.array([x for x in Time_100mg if ~np.isnan(x)])
Conc_100mg =  LidoDat['Conc100mg']*1e-6
Conc_100mg = np.array([x for x in Conc_100mg if ~np.isnan(x)])

## Partitioning parameters 
GA = 0
TIA = 0
params = lmfit.Parameters()
params.add('R', value = 1)
params.add('k_er', value = 100)
params.add('fu', value= 0.1)
 
## Define ODE's - Creates a 16x1 vector, U, of ODE's 
def dUdt(U,tspan, params):
    R = params['R'].value
    k_er = params['k_er'].value
    fu = params['fu'].value
    dudt = [1/pars.Vve*((pars.Qbr*U[3]*R)/(fu*pars.Kbr)+\
        (pars.Qh*U[4]*R)/(fu*pars.Kh)+(pars.Qmu*U[5]*R)/(fu*pars.Kmu)+(pars.Qadi*U[6]*R)/(fu*pars.Kadi)+\
        (pars.Qsk*U[7]*R)/(fu*pars.Ksk)+(pars.Qbo*U[8]*R)/(fu*pars.Kbo)+(pars.Qthy*U[9]*R)/(fu*pars.Kthy)+\
        (pars.Qk*U[10]*R)/(fu*pars.Kk)+(pars.Qli*U[15]*R)/(fu*pars.Kli)-pars.Qlu*U[0]+pars.VIR),# arterial
    1/pars.Vart*((pars.Qlu*R*U[2])/(fu*pars.Klu)-(pars.Qh+pars.Qbr+pars.Qmu+pars.Qadi+pars.Qsk+pars.Qsp\
        +pars.Qpa+pars.Qha+pars.Qst+pars.Qgut+pars.Qbo+pars.Qk+pars.Qthy)*U[1]+pars.AIR),# venus
    pars.Qlu/pars.Vlu*(U[0]-(U[2]*R)/(fu*pars.Klu)), #lungs
    pars.Qbr/pars.Vbr*(U[1]-(U[3]*R)/(fu*pars.Kbr)),#brain 
    pars.Qh/pars.Vh*(U[1]-(U[4]*R)/(fu*pars.Kh)), #heart
    pars.Qmu/pars.Vmu*(U[1]-(U[5]*R)/(fu*pars.Kmu)), #muscle
    pars.Qadi/pars.Vadi*(U[1]-(U[6]*R)/(fu*pars.Kadi)), #adipose
    pars.Qsk/pars.Vsk*(U[1]-(U[7]*R)/(fu*pars.Ksk)), # skin
    pars.Qbo/pars.Vbo*(U[1]-(U[8]*R)/(fu*pars.Kbo)), #bone
    pars.Qthy/pars.Vthy*(U[1]-(U[9]*R)/(fu*pars.Kthy)), # thymus 
    1/pars.Vk*(pars.Qk*(U[1]-(U[10]*R)/(fu*pars.Kk)))-U[10]*k_er/pars.Kk, #kidneys 
    pars.Qsp/pars.Vsp*(U[1]-(U[11]*R)/(fu*pars.Ksp)), # spleen
    pars.Qpa/pars.Vpa*(U[1]-(U[12]*R)/(fu*pars.Kpa)), # pancreas
    1/pars.Vst*(pars.Qst*(U[1]-(U[13]*R)/(fu*pars.Kst))+GA), #stomach 
    1/pars.Vgut*(pars.Qgut*(U[1]-(U[14]*R)/(fu*pars.Kgut))+TIA), # gut 
        1/pars.Vli*(pars.Qha*U[1]+(pars.Qgut*U[14]*R)/(fu*pars.Kgut)+(pars.Qpa*U[12]*R)/(fu*pars.Kpa)+\
        (pars.Qsp*U[11]*R)/(fu*pars.Ksp)+(pars.Qst*U[13]*R)/(fu*pars.Kst)-(pars.Qli*U[15]*R)/(fu*pars.Kli)\
    -(U[15]*pars.Vmax)/(pars.Km*pars.Kli+1e-10))] #liver
    return dudt

# Run the ODE's for 3 different ICS, copare to data, return some error metric
def resid(params):
    ICS = np.zeros([16,3]) # Initial conditions
    ICS[0,:] = Dose/pars.Vve # initial IA dose 
    sol25 = odeint(dUdt, ICS[:,0], tspan, args=(params,))
    sol75 = odeint(dUdt, ICS[:,1], tspan, args=(params,))
    sol100 = odeint(dUdt, ICS[:,2], tspan, args=(params,))
    Err_25 = sum((Conc_25mg-sol25[pars.col25,0])**2)
    Err_75 = sum((Conc_75mg-sol75[pars.col75,0])**2)
    Err_100 = sum((Conc_100mg-sol100[pars.col100,0])**2)
    return Err_25+Err_75+Err_100

# Fit using NlinLSQ

Fit = lmfit.minimize(resid, params, args=None, method='nelder')
print("# Fit using leastsq:")
lmfit.report_fit(Fit)

## Plot results
params.add('R', Fit.x[0])
params.add('k_er', Fit.x[1])
params.add('fu', Fit.x[2])
ICS = np.zeros([16,3]) # Initial conditions
ICS[0,:] = Dose/pars.Vve # initial IA dose 
FinalSol25 = odeint(dUdt, ICS[:,0], tspan, args=(params,))
FinalSol75 = odeint(dUdt, ICS[:,1], tspan, args=(params,))
FinalSol100 = odeint(dUdt, ICS[:,2], tspan, args=(params,))
 
plt.semilogy(tspan,FinalSol25[:,0],'r--',Time_25mg,Conc_25mg,'ro',tspan,FinalSol75[:,0],'b--',\
             Time_75mg,Conc_75mg,'bo',tspan,FinalSol100[:,0],'k--',Time_25mg,Conc_100mg,'ko')
plt.legend(['25mg','', '75mg','', '100mg',''])
plt.xlabel('Time, (mins)')
plt.ylabel('Plasma concentration, ng/mL')
plt.show()
