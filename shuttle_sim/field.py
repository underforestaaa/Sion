"""Sion surface-electrode backend. Public positions/derivatives are in SI."""
from dataclasses import dataclass
import numpy as np
from scipy.constants import elementary_charge, atomic_mass
from scipy.optimize import least_squares, minimize


class Rejected(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


def finite(value, name):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite')
    return a


@dataclass
class Local:
    position_m: np.ndarray
    hessian_J_m2: np.ndarray
    omega_rad_s: np.ndarray
    modes: np.ndarray  # columns, ascending eigenfrequency (not persistent mode labels)


class SionTrap:
    """Wrap a given electrode.System; RF values must already be Sion-scaled.

    One RF frequency, common RF phase, positive ion; pseudopotential only.
    The backend is mutable and this adapter is intentionally NOT thread safe.
    """
    def __init__(self, system, length_m, mass_kg, charge_C, dc_indices, lower_V, upper_V):
        self.system = system
        self.L, self.mass, self.q = map(float, (length_m, mass_kg, charge_C))
        finite([self.L, self.mass, self.q], 'scales')
        if min(self.L, self.mass, self.q) <= 0:
            raise ValueError('positive length, mass, charge required')
        self.indices = np.asarray(dc_indices, int)
        if len(set(self.indices)) != len(self.indices) or not len(self.indices):
            raise ValueError('unique nonempty DC indices required')
        if np.any(self.indices < 0) or np.any(self.indices >= len(system)):
            raise ValueError('DC index outside system')
        self.names = [system[i].name for i in self.indices]
        self.lo = np.broadcast_to(finite(lower_V, 'lower_V'), (len(self.indices),)).copy()
        self.hi = np.broadcast_to(finite(upper_V, 'upper_V'), self.lo.shape).copy()
        if np.any(self.lo >= self.hi):
            raise ValueError('invalid voltage bounds')
        self.fixed = system.dcs.copy()
        self.fixed[self.indices] = 0

    def basis(self, r_m, derivative):
        from sion.electrode.utils import expand_tensor
        r = finite(r_m, 'position_m') / self.L
        b = self.system.individual_potential(r, derivative)
        if derivative:
            b = expand_tensor(b)
        return b[self.indices, 0] * self.q / self.L**derivative

    def rf(self, r_m, derivative):
        with self.system.with_voltages(dcs=self.fixed):
            p = self.system.potential(finite(r_m, 'position_m') / self.L, derivative)[0]
        return p * self.q / self.L**derivative

    def energy_derivative(self, r_m, voltage_V, derivative=0, stray_V_m=None):
        v = finite(voltage_V, 'voltage_V')
        if v.shape != self.lo.shape:
            raise ValueError('voltage channel shape mismatch')
        p = self.rf(r_m, derivative) + np.tensordot(v, self.basis(r_m, derivative), axes=1)
        if stray_V_m is not None:
            e = finite(stray_V_m, 'stray field')
            if e.shape!=(3,):
                raise ValueError('stray field must have three components')
            if derivative == 0:
                p -= self.q * np.dot(e, r_m)
            elif derivative == 1:
                p -= self.q * e
        return float(np.asarray(p).item()) if derivative == 0 else p

    def local(self, voltage_V, seed_m, stray_V_m=None):
        seed = finite(seed_m, 'seed_m')
        def fun(y):
            return self.energy_derivative(y*self.L, voltage_V, 1, stray_V_m)*self.L/self.q
        def jac(y):
            return self.energy_derivative(y*self.L, voltage_V, 2)*self.L**2/self.q
        y0 = seed/self.L
        if y0[2] <= .01:
            raise Rejected('no_minimum', 'seed below surface')
        sol = least_squares(lambda y:1e4*fun(y), y0, jac=lambda y:1e4*jac(y), bounds=([-np.inf,-np.inf,.01], [np.inf]*3),
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=100)
        r = sol.x*self.L
        H = self.energy_derivative(r, voltage_V, 2)
        eig, modes = np.linalg.eigh(H/self.mass)
        if not sol.success or np.linalg.norm(fun(sol.x)) > 1e-8:
            raise Rejected('no_minimum', sol.message)
        if eig.min() <= 0:
            raise Rejected('weak_confinement', 'nonpositive curvature')
        return Local(r, H, np.sqrt(eig), modes)

    def invert(self, target_m, axial_direction, axial_omega_rad_s, previous_V, regularization=1e-4):
        """Hard 3D force balance + one axial curvature; radial modes remain free.

        Minimum-change solution uses null-space freedom with temporal regularization.
        Full Hessian is NOT prescribed: DC Hessians obey Laplace's equation.
        """
        r = finite(target_m, 'target_m'); a = finite(axial_direction, 'axis')
        a = a/np.linalg.norm(a)
        w = float(axial_omega_rad_s)
        if not np.isfinite(w) or w <= 0 or regularization <= 0:
            raise ValueError('positive finite frequency and regularization required')
        G, H = self.basis(r, 1), self.basis(r, 2)
        A = np.vstack([G.T*self.L/self.q,
                       np.einsum('i,cij,j->c', a,H,a)*self.L**2/self.q])
        b = np.r_[-self.rf(r,1)*self.L/self.q,
                  (self.mass*w*w-a@self.rf(r,2)@a)*self.L**2/self.q]
        scale = np.maximum(np.linalg.norm(A,axis=1), 1e-15)
        A, b = A/scale[:,None], b/scale
        prev = np.clip(finite(previous_V,'previous_V'),self.lo,self.hi)
        res = minimize(lambda v: .5*np.sum((v-prev)**2)+.5*regularization*np.sum(v*v), prev,
                       jac=lambda v: v-prev+regularization*v,
                       constraints=[{'type':'eq','fun':lambda v:A@v-b,'jac':lambda v:A}],
                       bounds=list(zip(self.lo,self.hi)),method='SLSQP',
                       options={'ftol':1e-11,'maxiter':150})
        if not res.success or np.max(np.abs(A@res.x-b)) > 1e-6:
            raise Rejected('infeasible_voltages', res.message)
        local = self.local(res.x,r)
        if np.linalg.norm(local.position_m-r) > 5e-9:
            raise Rejected('infeasible_voltages','minimum misses target')
        return res.x, local


def load_trap(cfg):
    import sion
    mass = cfg['mass_amu']*atomic_mass
    charge = cfg['charge_e']*elementary_charge
    L = cfg['length_scale_m']
    finite([mass,charge,L,cfg['rf_frequency_hz']], 'trap scales')
    if min(mass,charge,L,cfg['rf_frequency_hz']) <= 0:
        raise ValueError('positive trap scales required')
    O = 2*np.pi*cfg['rf_frequency_hz']
    if cfg['kind'] == 'sion_five_wire':
        d = cfg['geometry']
        amp = cfg['rf_peak_V']*np.sqrt(charge/mass)/(2*L*O)
        sys = sion.five_wire_trap_design(amp,d['top_dc'],d['bottom_dc'],d['top_rf'],
                d['bottom_rf'],d['central_dc'],L=L,need_plot=False)
    elif cfg['kind'] == 'gds':
        # No implicit polygon order: fabrication layers are not channel IDs.
        from gdspy import GdsLibrary
        from sion.electrode import System, PolygonPixelElectrode
        from sion.electrode.utils import area_centroid
        lib = GdsLibrary(infile=cfg['path'])
        cells = lib.top_level()
        if cfg.get('cell'):
            cells = [lib.cells[cfg['cell']]]
        if len(cells) != 1:
            raise ValueError('select exactly one GDS top-level cell')
        polygons = cells[0].get_polygons(by_spec=True)
        mapping = cfg.get('electrode_map')
        if not mapping:
            raise ValueError('GDS requires explicit electrode_map: name, layer, datatype')
        electrodes=[]; used=set()
        for entry in mapping:
            spec=(int(entry['layer']),int(entry.get('datatype',0)))
            if spec in used:
                raise ValueError('GDS layer/datatype assigned twice')
            used.add(spec)
            paths=[]
            for poly in polygons.get(spec,[]):
                # gdspy coordinates are in library user units, not necessarily um.
                poly=finite(poly,'GDS polygon')*(lib.unit/L)
                area,_=area_centroid(poly)
                if abs(area)<1e-18:
                    raise ValueError('degenerate GDS polygon')
                # Positive physical unit-voltage potential: CCW viewed from above.
                paths.append(poly[::-1] if area<0 else poly)
            if not paths:
                raise ValueError('mapped GDS electrode has no polygons')
            electrodes.append(PolygonPixelElectrode(name=entry['name'],paths=paths))
        if len({e.name for e in electrodes})!=len(electrodes):
            raise ValueError('unique electrode names required')
        sys=System(electrodes)
        rf=np.zeros(len(sys))
        for key,peak in cfg['rf_electrodes'].items():
            names=[e.name for e in sys]
            index=names.index(key) if key in names else int(key)
            rf[index]=float(peak)*np.sqrt(charge/mass)/(2*L*O)
        sys.rfs=rf
    else:
        raise ValueError('kind must be sion_five_wire or gds; for an existing System use SionTrap')
    if not np.isfinite(sys.rfs).all():
        raise ValueError('RF amplitudes must be finite')
    if any(abs(sys[i].rf)>0 for i in cfg['dc_indices']):
        raise ValueError('RF electrodes must not overlap independently driven DC channels')
    adapter = SionTrap(sys,L,mass,charge,cfg['dc_indices'],cfg['lower_V'],cfg['upper_V'])
    return adapter
