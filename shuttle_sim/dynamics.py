"""3D Gaussian evolution: exact piecewise affine vs adaptive ODE.

No RF micromotion, stochastic heating or non-Gaussian states.
Internal scaling: L=1 um, T=1 us; public inputs/outputs are SI.
"""
import numpy as np
from scipy.constants import hbar
from scipy.integrate import solve_ivp
from .field import Rejected, finite

L=1e-6
T=1e-6


def initial_state(local, mass, nbar=0.):
    if not np.isfinite(mass) or mass<=0 or not np.isfinite(nbar) or nbar<0:
        raise ValueError('positive mass and nonnegative finite nbar required')
    w=finite(local.omega_rad_s,'omega_rad_s');R=finite(local.modes,'modes')
    if w.shape!=(3,) or np.any(w<=0) or R.shape!=(3,3) or not np.allclose(R.T@R,np.eye(3),atol=1e-10):
        raise ValueError('three positive frequencies and orthonormal modes required')
    C=np.zeros((6,6))
    C[:3,:3]=R@np.diag((2*nbar+1)*hbar/(2*mass*w))@R.T/L**2
    C[3:,3:]=R@np.diag((2*nbar+1)*hbar*w/(2*mass))@R.T*(T/L)**2
    return np.r_[local.position_m/L,np.zeros(3)],C


def summarize(z,C,local,mass,target_m):
    C=(C+C.T)/2
    if not np.isfinite(z).all() or not np.isfinite(C).all():
        raise Rejected('numerical_failure','nonfinite Gaussian state')
    if np.linalg.eigvalsh(C).min() < -1e-12:
        raise Rejected('numerical_failure','negative covariance')
    dx=z[:3]*L-local.position_m;v=z[3:]*L/T
    R=local.modes;w=local.omega_rad_s
    X=R.T@(C[:3,:3]*L**2+np.outer(dx,dx))@R
    V=R.T@(C[3:,3:]*(L/T)**2+np.outer(v,v))@R
    raw=mass*(np.diag(V)+w*w*np.diag(X))/(2*hbar*w)-.5
    if np.min(raw)<-1e-5:
        raise Rejected('numerical_failure','unphysical negative excitation')
    n=np.maximum(raw,0)
    J=np.block([[np.zeros((3,3)),np.eye(3)],[-np.eye(3),np.zeros((3,3))]])
    quantum_half=hbar*T/(2*mass*L**2)
    if np.linalg.eigvalsh(C+1j*quantum_half*J).min() < -1e-7*quantum_half:
        raise Rejected('numerical_failure','covariance violates uncertainty relation')
    sym=np.linalg.eigvals(1j*J@C)
    Xcov=R.T@(C[:3,:3]*L**2)@R
    Vcov=R.T@(C[3:,3:]*(L/T)**2)@R
    nc=mass*(np.diag(Vcov)+w*w*np.diag(Xcov))/(2*hbar*w)-.5
    coherent=mass*((R.T@v)**2+w*w*(R.T@dx)**2)/(2*hbar*w)
    return {'n_modes':n.tolist(),'n_total':float(n.sum()),
            'coherent_n_modes':coherent.tolist(),'coherent_n_total':float(coherent.sum()),
            'covariance_n_modes':nc.tolist(),'covariance_n_total':float(nc.sum()),
            'mean_position_m':(z[:3]*L).tolist(),'mean_velocity_m_s':v.tolist(),
            'position_error_m':float(np.linalg.norm(z[:3]*L-target_m)),
            'well_target_error_m':float(np.linalg.norm(local.position_m-target_m)),
            'phase_space_displacement_m':dx.tolist(),
            'covariance_dimensionless':C.tolist(),'covariance_min_eigenvalue':float(np.linalg.eigvalsh(C).min()),
            'min_symplectic_eigenvalue_over_hbar_half':float(np.min(np.abs(sym))/(hbar*T/(2*mass*L**2)))}


def propagate(times_s, centers_m, hessians_J_m2, initial, final, mass, target_m, method='exact',hold='linear'):
    ts=finite(times_s,'time');r=finite(centers_m,'centers');H=finite(hessians_J_m2,'H')
    if hold not in ('linear','zero_order'):
        raise ValueError('unknown hold rule')
    if ts.ndim!=1 or len(ts)<2 or r.shape!=(len(ts),3) or H.shape!=(len(ts),3,3) or np.any(np.diff(ts)<=0):
        raise ValueError('inconsistent dynamics grid')
    if not np.allclose(H,H.transpose(0,2,1),rtol=1e-12,atol=0):
        raise ValueError('Hessians must be symmetric')
    z,C=initial_state(initial,mass)
    K=H/mass*T*T;c=r/L;tt=ts/T
    if np.min(np.linalg.eigvalsh(K))<=0:
        raise Rejected('weak_confinement','ROM Hessian not positive')
    if method=='exact':
        for k,dt in enumerate(np.diff(tt)):
            km=K[k] if hold=='zero_order' else (K[k]+K[k+1])/2
            velocity=np.zeros(3) if hold=='zero_order' else (c[k+1]-c[k])/dt
            eig,R=np.linalg.eigh(km);w=np.sqrt(eig)
            co=R@np.diag(np.cos(w*dt))@R.T
            si=R@np.diag(np.sin(w*dt)/w)@R.T
            S=np.block([[co,si],[-km@si,co]])
            # Exact affine solution for constant K and a linearly moving minimum.
            eq0=np.r_[c[k],velocity]
            eq1=np.r_[c[k] if hold=='zero_order' else c[k+1],velocity]
            z=eq1+S@(z-eq0);C=S@C@S.T
    elif method=='dop853' and hold=='zero_order':
        for k,dt in enumerate(np.diff(tt)):
            A=np.block([[np.zeros((3,3)),np.eye(3)],[-K[k],np.zeros((3,3))]])
            b=np.r_[np.zeros(3),K[k]@c[k]]
            def rhs(t,y):
                cov=y[6:].reshape(6,6)
                return np.r_[A@y[:6]+b,(A@cov+cov@A.T).ravel()]
            sol=solve_ivp(rhs,(0,dt),np.r_[z,C.ravel()],method='DOP853',rtol=2e-10,atol=2e-13)
            if not sol.success: raise Rejected('numerical_failure',sol.message)
            z=sol.y[:6,-1];C=sol.y[6:,-1].reshape(6,6)
    elif method=='dop853':
        def rhs(t,y):
            k=min(max(np.searchsorted(tt,t)-1,0),len(tt)-2)
            a=np.clip((t-tt[k])/(tt[k+1]-tt[k]),0,1)
            km=(1-a)*K[k]+a*K[k+1];rm=(1-a)*c[k]+a*c[k+1]
            A=np.block([[np.zeros((3,3)),np.eye(3)],[-km,np.zeros((3,3))]])
            yy=y[:6];cov=y[6:].reshape(6,6)
            return np.r_[A@yy+np.r_[np.zeros(3),km@rm],(A@cov+cov@A.T).ravel()]
        sol=solve_ivp(rhs,(tt[0],tt[-1]),np.r_[z,C.ravel()],method='DOP853',
                      rtol=2e-10,atol=2e-13,max_step=float(np.min(np.diff(tt))))
        if not sol.success:
            raise Rejected('numerical_failure',sol.message)
        z=sol.y[:6,-1];C=sol.y[6:,-1].reshape(6,6)
    else:
        raise ValueError('unknown propagator')
    return summarize(z,C,final,mass,target_m)


def validate_full_field_centroid(trap,times_s,voltages,initial,target_m,stray=None,frequency_scale=1.):
    """Nonlinear pseudopotential force; reports coherent excitation only, not total n."""
    if not np.isfinite(frequency_scale) or frequency_scale<=0:
        raise ValueError('positive frequency scale required')
    t=finite(times_s,'time')/T;V=finite(voltages,'voltages');z0=np.r_[initial.position_m/L,np.zeros(3)]
    def rhs(tau,z):
        v=np.array([np.interp(tau,t,V[:,j]) for j in range(V.shape[1])])
        a=-frequency_scale**2*trap.energy_derivative(z[:3]*L,v,1,stray)/trap.mass*T*T/L
        return np.r_[z[3:],a]
    sol=solve_ivp(rhs,(t[0],t[-1]),z0,method='DOP853',rtol=2e-9,atol=1e-10,
                  max_step=min(float(np.min(np.diff(t))),np.pi/(np.max(initial.omega_rad_s)*T)))
    if not sol.success: raise Rejected('numerical_failure',sol.message)
    final=trap.local(V[-1],target_m,stray);z=sol.y[:,-1]
    omega=final.omega_rad_s*frequency_scale
    dx=final.modes.T@(z[:3]*L-final.position_m);vv=final.modes.T@(z[3:]*L/T)
    n=trap.mass*(vv*vv+omega**2*dx*dx)/(2*hbar*omega)
    return {'coherent_n_modes':n.tolist(),'coherent_n_total':float(n.sum()),
            'mean_position_m':(z[:3]*L).tolist(),'mean_velocity_m_s':(z[3:]*L/T).tolist(),
            'position_error_m':float(np.linalg.norm(z[:3]*L-target_m)),
            'force_evaluations':sol.nfev,'model':'full Sion pseudopotential centroid; NOT full quantum n'}
