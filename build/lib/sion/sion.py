from pylion.lammps import lammps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from electrode import (System, PolygonPixelElectrode, PointPixelElectrode)
import sys
from scipy.optimize import (minimize, curve_fit)
import scipy.constants as ct
import gdspy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
import warnings


ech = 1.602176634e-19  # electron charge, C
amu = 1.66053906660e-27  # atomic mass unit, kg
eps0 = 8.8541878128e-12  # vacuum electric permittivity

"""
Functions, simulating the ion dynamics above planar traps
"""

@lammps.fix
def polygon_trap(uid, Omega, rf_voltages, dc_voltages, RFs, DCs, cover=(0, 0)):
    """
    Simulates an arbitrary planar trap with polygonal electrodes. 
    Everything must be in standard units

    Parameters
    ----------
    uid : str
        ions ID, .self parameter
    Omega : float or list shape len(RFs)
        array of RF frequencies (or a frequency if they are the same) of each RF electrode
    rf_voltages : array shape (len(RFs))
        Set of the peak voltages on RF electrodes.
    dc_voltages : array shape (len(DCs))
        Set of the voltages on DC electrodes.
    RFs : list shape (len(RFs), 4)
        Array of coordinates of RF electrodes in m
    DCs : list shape (len(DCs), 4)
        Array of coordinates of DC electrodes in m
        The order of electrodes must match the order of voltage set and omegas.
    cover : list shape (2), optional
        array [cover_number, cover_height] - number of the terms in
        cover electrode influence expansion (5 is mostly enough) and its height.

    Returns
    -------
    odict : str
        updates simulation lines, which will be executed by LAMMPS
    """
    odict = {}
    rf = [len(a) for a in RFs]
    nrf = len(rf)
    dc = [len(a) for a in DCs]
    ndc = len(dc)
    try:
        Omega[0]
    except:
        Omega = np.ones(nrf)*Omega
    Omega = np.array(Omega)
    lines = [
        f'\n# Creating a polygonal Surface Electrode Trap... (fixID={uid})']
    odict['timestep'] = 1 / np.max(Omega) / 20
    for i in range(nrf):
        lines.append(
            f'variable phase{uid}{i:d}\t\tequal "{Omega[i]:e}*step*dt"')
    xcc = []
    ycc = []
    zcc = []
    u_set = np.concatenate((np.array(rf_voltages), np.array(dc_voltages)))
    #necessary for correct computation
    u_set = -u_set

    for iterr in range(nrf):
        xc = []
        yc = []
        zc = []
        polygon = np.array(RFs[iterr])
        no = np.array(polygon).shape[0]

        for m in range(2*cover[0]+1):
            xt = f'(x - ({polygon[no-1, 0]:e}))'
            numt = no-1
            yt = f'(y - ({polygon[no-1, 1]:e}))'
            cov = 2*(m - cover[0])*cover[1]
            if cover[0] == 0:
                z = 'z'
            else:
                z = f'(z + ({cov:e}))'

            for k in range(no):
                xo = xt
                yo = yt
                numo = numt
                numt = k
                dx = polygon[numt, 0] - polygon[numo, 0]
                dy = polygon[numo, 1] - polygon[numt, 1]
                c = polygon[numt, 0] * polygon[numo, 1] - polygon[numo, 0] * polygon[numt, 1]
                lt = (polygon[numt, 0] - polygon[numo, 0]) ** 2 + (polygon[numt, 1] - polygon[numo, 1]) ** 2
                ro = f'sqrt({xo}^2+{yo}^2+{z}^2)'
                xt = f'(x - ({polygon[k, 0]:e}))'

                yt = f'(y - ({polygon[k, 1]:e}))'
                rt = f'sqrt({xt}^2+{yt}^2+{z}^2)'
                
                if u_set[iterr] != 0:
                    n = f'({ro}+{rt})/({ro}*{rt}*(({ro}+{rt})*({ro}+{rt})-({lt:e})))'
                    if dx != 0:
                        yc.append(
                            f'({dx:e})*{z}*{n}')
                    if dy != 0:
                        xc.append(
                            f'({dy:e})*{z}*{n}')
                    if (c**2 + dx**2 + dy**2 == 0):
                        pass
                    elif (c**2 + dx**2 == 0):
                        zc.append(f'(-({dy:e})*x)*{n}')
                    elif (c**2 + dy**2 == 0):
                        zc.append(f'(-({dx:e})*y)*{n}')
                    elif (dx**2 + dy**2 == 0):
                        zc.append(f'({c:e})*{n}')
                    elif (c == 0):
                        zc.append(f'(- ({dx:e})*y - ({dy:e})*x)*{n}')
                    elif (dx == 0):
                        zc.append(f'({c:e} - ({dy:e})*x)*{n}')
                    elif (dy == 0):
                        zc.append(f'({c:e} - ({dx:e})*y)*{n}')
                    else:
                        zc.append(f'({c:e} - ({dx:e})*y - ({dy:e})*x)*{n}')
        
        xc = ' + '.join(xc)
        yc = ' + '.join(yc)
        zc = ' + '.join(zc)
        
        xcc.append(f'({u_set[iterr]/np.pi:e})*({xc})*cos(v_phase{uid}{iterr:d})')
        ycc.append(f'({u_set[iterr]/np.pi:e})*({yc})*cos(v_phase{uid}{iterr:d})')
        zcc.append(f'({u_set[iterr]/np.pi:e})*({zc})*cos(v_phase{uid}{iterr:d})')
    xcc = ' + '.join(xcc)
    ycc = ' + '.join(ycc)
    zcc = ' + '.join(zcc)    

    xr = []
    yr = []
    zr = []
    for ite in range(ndc):
        polygon = np.array(DCs[ite])
        no = np.array(polygon).shape[0]

        for m in range(2 * cover[0] + 1):
            x2 = f'(x - ({polygon[no-1, 0]:e}))'
            y2 = f'(y - ({polygon[no-1, 1]:e}))'
            numt = no - 1
            cov = 2 * (m - cover[0]) * cover[1]
            z = f'(z + ({cov:e}))'

            for k in range(no):
                numo = numt
                numt = k
                rodc = f'sqrt({x2}^2+{y2}^2+{z}^2)'

                x2 = f'(x - ({polygon[k, 0]:e}))'
                y2 = f'(y - ({polygon[k, 1]:e}))'
                rtdc = f'sqrt({x2}^2+{y2}^2+{z}^2)'

                dx = polygon[numt, 0] - polygon[numo, 0]
                dy = polygon[numo, 1] - polygon[numt, 1]
                c = polygon[numt, 0] * polygon[numo, 1] - polygon[numo, 0] * polygon[numt, 1]
                lt = (polygon[numt, 0] - polygon[numo, 0]) ** 2 + (polygon[numt, 1] - polygon[numo, 1]) ** 2
                if u_set[ite+nrf] != 0:
                    n = f'({u_set[ite+nrf]:e})*({rodc}+{rtdc})/({rodc}*{rtdc}*(({rodc}+{rtdc})*({rodc}+{rtdc})-({lt:e}))*{np.pi:e})'

                    if dx != 0:
                        yr.append(
                            f'({dx:e})*{z}*{n}')
                    if dy != 0:
                        xr.append(
                            f'({dy:e})*{z}*{n}')
                    if (c**2 + dx**2 + dy**2 == 0):
                        pass
                    elif (c**2 + dx**2 == 0):
                        zr.append(f'(-({dy:e})*x)*{n}')
                    elif (c**2 + dy**2 == 0):
                        zr.append(f'(-({dx:e})*y)*{n}')
                    elif (dx**2 + dy**2 == 0):
                        zr.append(f'({c:e})*{n}')
                    elif (c == 0):
                        zr.append(f'(- ({dx:e})*y - ({dy:e})*x)*{n}')
                    elif (dx == 0):
                        zr.append(f'({c:e} - ({dy:e})*x)*{n}')
                    elif (dy == 0):
                        zr.append(f'({c:e} - ({dx:e})*y)*{n}')
                    else:
                        zr.append(f'({c:e} - ({dx:e})*y - ({dy:e})*x)*{n}')

    xr = ' + '.join(xr)
    yr = ' + '.join(yr)
    zr = ' + '.join(zr)
    
    if len(xr) == 0:
        xr = '0'
        yr = '0'
        zr = '0'

    lines.append(f'variable oscEX{uid} atom "{xcc}+{xr}"')
    lines.append(f'variable oscEY{uid} atom "{ycc}+{yr}"')
    lines.append(f'variable oscEZ{uid} atom "{zcc}+{zr}"')
    lines.append(
        f'fix {uid} all efield v_oscEX{uid} v_oscEY{uid} v_oscEZ{uid}\n')

    odict.update({'code': lines})

    return odict

@lammps.fix
def point_trap(uid, trap, cover=(0, 0)):
    """
    Simulates an arbitrary point trap. The point trap means a trap of 
    an arbitrary shape, which is approximated by circle-shaped electrodes, 
    called points. The points potential is approximated from the fact, that it 
    has infinitesemal radius, so the smaller are points, the more precise is
    the simulation (but slower).
    
    Parameters
    ----------
    trap : list
        List of two elements: [trap_rf, trap_dc], obtained from point_trap_design()
        trap_rf: lits of rf electrodes, each contains 4 parameters: 
            [frequency: RF frequency of the electrode,
             voltage: peak voltage of the electrode, 
             centers: coordinates of points' centers, 
             areas: areas of each point]
        trap_dc: lits of dc electrodes (can be empty), each contains 4 parameters: 
            [voltage: voltage of the electrode,
             centers: coordinates of points' centers, 
             areas: areas of each point]
    cover : [cover_max, cover_height], optional
    
    Returns
    -------
    odict : str
        Updates of simulation with point trap

    """
    odict = {}
    lines = [f'\n# Creating a point Surface Electrode Trap... (fixID={uid})']
    Omega = []
    for elec in trap[0]:
        Omega.append(elec[0])
    Omega = np.array(Omega)
    odict['timestep'] = 1 / np.max(Omega) / 20
    n_rf_elecs = len(Omega)
    for i in range(n_rf_elecs):
        lines.append(
            f'variable phase{uid}{i:d}\t\tequal "{Omega[i]:e}*step*dt"')
    xrr = []
    yrr = []
    zrr = []
    for num, elec in enumerate(trap[0]):
        xr = []
        yr = []
        zr = []
        for i in range(len(elec[2])):
            for m in range(2*cover[0]+1):
                xt = f'(x - ({elec[3][i][0]:e}))'
                yt = f'(y - ({elec[3][i][1]:e}))'
                cov = 2 * (m - cover[0]) * cover[1]
                if cov == 0:
                    z = 'z'
                else:
                    z = f'(z + ({cov:e}))'
                r = f'sqrt({xt}^2+{yt}^2+{z}^2)'

                xr.append(
                    f'(-3*{xt}*{z}/{r}^5)')
                yr.append(
                    f'(-3*{yt}*{z}/{r}^5)')
                zr.append(
                    f'({xt}^2 + {yt}^2 - 2*{z}^2)/{r}^5')
            
        xr = ' + '.join(xr)
        yr = ' + '.join(yr)
        zr = ' + '.join(zr)
        
        if elec[1] != 0:
            xrr.append(f'({elec[2][0]*elec[1]/2/np.pi:e})*({xr})*cos(v_phase{uid}{num:d})')
            yrr.append(f'({elec[2][0]*elec[1]/2/np.pi:e})*({yr})*cos(v_phase{uid}{num:d})')
            zrr.append(f'({elec[2][0]*elec[1]/2/np.pi:e})*({zr})*cos(v_phase{uid}{num:d})')
    
    if len(xrr) > 0:
        xrr = ' + '.join(xrr)
        yrr = ' + '.join(yrr)
        zrr = ' + '.join(zrr)
    else:
        xrr = '0'
        yrr = '0'
        zrr = '0'

    xcc = []
    ycc = []
    zcc = []
    for num, elec in enumerate(trap[1]):
        xc = []
        yc = []
        zc = []
        for i in range(len(elec[1])):
            for m in range(2*cover[0]+1):
                xt = f'(x - ({elec[2][i][0]:e}))'
                yt = f'(y - ({elec[2][i][1]:e}))'
                cov = 2 * (m - cover[0]) * cover[1]
                if cov == 0:
                    z = 'z'
                else:
                    z = f'(z + ({cov:e}))'
                r = f'sqrt({xt}^2+{yt}^2+{z}^2)'

                xc.append(
                    f'(-3*{xt}*{z}/{r}^5)')
                yc.append(
                    f'(-3*{yt}*{z}/{r}^5)')
                zc.append(
                    f'({xt}^2 + {yt}^2 - 2*{z}^2)/{r}^5')
            
        xc = ' + '.join(xc)
        yc = ' + '.join(yc)
        zc = ' + '.join(zc)
        
        if elec[0] != 0:
            xcc.append(f'({elec[1][0]*elec[0]/2/np.pi:e})*({xc})')
            ycc.append(f'({elec[1][0]*elec[0]/2/np.pi:e})*({yc})')
            zcc.append(f'({elec[1][0]*elec[0]/2/np.pi:e})*({zc})')
    
    if len(xcc) > 0:
        xcc = ' + '.join(xcc)
        ycc = ' + '.join(ycc)
        zcc = ' + '.join(zcc)
    else:
        xcc = '0'
        ycc = '0'
        zcc = '0'

    lines.append(f'variable oscEX{uid} atom "{xrr}+{xcc}"')
    lines.append(f'variable oscEY{uid} atom "{yrr}+{ycc}"')
    lines.append(f'variable oscEZ{uid} atom "{zrr}+{zcc}"')
    lines.append(
        f'fix {uid} all efield v_oscEX{uid} v_oscEY{uid} v_oscEZ{uid}\n')

    odict.update({'code': lines})

    return odict



@lammps.fix
def ringtrap(uid, Omega, r, R, r_dc, RF_voltage, dc_voltage, resolution=100, cover=(0, 0)):
    """
    Simulates a ring-shaped RF electrode with central DC electrode
    
    Parameters
    ----------
    Omega : float
        RF frequency of the ring
    r : float
        inner radius of the ring
    R : float
        outer radius of the ring
    r_dc : float
        radius of central dc electrode (can be 0)
    RF_voltage : float
        RF amplitude of the ring
    DC_voltage : float
        DC voltage of the ring
    resolution : int, optional
        100 is very high resolution.
        number of points in the ring design
    cover : [cover_max, cover_height], optional
        
    Returns
    -------
    odict : str
        updates simulation of a single ring

    """
    s, omega, trap, volt = ring_trap_design(RF_voltage, Omega, r, R, r_dc, dc_voltage, res = resolution, need_coordinates = True, cheight=cover[0], cmax=cover[1])

    return point_trap(trap, cover)


"""
Shuttling of ions in the trap
"""

def linear_shuttling_voltage(s, x0, d, T, dc_set, shuttlers = 0, N = 4, vmin = -15, vmax = 15, res = 25, L = 1e-6, need_func = False):
    """
    Performs optimization of voltage sequence on DC electrodes for the linear
    shuttling, according to the tanh route. Voltage is optimized to maintain 
    axial secular frequency along the route.

    Parameters
    ----------
    s : electrode.System object
        Surface trap from electrode package
    x0 : array shape (3)
        Starting point of the shuttling
    d : float
        distance of the shuttling
    T : float
        Time of the shuttling operation
    dc_set : list shape (len(DCs))
        list of starting DC voltages
    shuttlers : list, optional, default is 0
        list of DC electrodes (defined as numbers), used for shuttling
        if ==0 then all dc electrodes participate in shuttling
        if ==[2,4,6] for example, only 2, 4 and 6th electrodes are participating,
        the rest are stationary dc
    N : int, optional
        4 means steep slope
        parameter in the tanh route definition (see q_tan)
    vmin : float, optional
        Minimal allowable DC voltage. The default is -15.
    vmax : float, optional
        Maximal allowable DC voltage. The default is 15.
    res : int, optional
        Number of steps of the voltage sequence during shuttling time.
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.
    need_func : bool, optional
        if True, the approximation functions of voltage sequences are provided, 
        which are used for MD simulation of shuttling

    Returns
    -------
    voltage_seq : list shape (len(shuttlers), res+1)
        voltage sequences on each shuttler electrode (could be constant)
    funcs : list of strings shape (len(DCs)), optional
        list of functions in format, used for MD simulation (including stationary DC)

    """
    def q_tan(t):
        return np.array([d/2*(np.tanh(N*(2*t-T)/T)/np.tanh(N) + 1), 0, 0])    
    
    return shuttling_voltage(s, x0, q_tan, T, dc_set, shuttlers, vmin, vmax, res, L, need_func, a = 1)
    
def fitter_tan(t, a, b, c, d):
    return a*np.tanh(b*(t + c)) + d

def fitter_norm(t, a, b, c, d):
    return a*np.exp(b*(t-c)**2) + d

def approx_linear_shuttling(voltage_seq, T, dc_set, shuttlers, res):
    """
    Approximation of voltage sequences on DC electrodes with analytic functions

    Parameters
    ----------
    voltage_seq : list shape (len(DCs), res+1)
        voltage sequences on each DC electrode (could be constant)
    T : float
        Time of shuttling operation
    dc_set : list shape (len(DCs))
        list of starting DC voltages
    shuttlers : list
        list of DC electrodes (defined as numbers), used for shuttling
    res : int, optional
        Resolution of the voltage sequence definition. The default is 50.

    Returns
    -------
    funcs : list of strings shape (len(DCs))
        List of functions, which are applied to the polygon_shuttling simulation
        Could be :
            constant - voltage of unchanging DC 
            a*tanh(b*t+c)+d - tanh curve
            a*exp(b*(t-c)^2)+d - normal distribution bell

    """
    x_data = np.arange(res+1)*T/res
    funcs = []
    
    for i, seq in enumerate(voltage_seq):
        mean = np.mean(seq)
        dif = seq[-1] - seq[0]
        ampl = np.max(seq)-np.min(seq)
        lin = np.std(seq)/np.max([mean, np.mean(np.abs(seq)), 1e-6])
        att = 0
        if lin < 1e-4:
            funcs.append('(%5.3f)' % mean)
        else:
            try:
                popt_tan, pcov_tan = curve_fit(fitter_tan, x_data, seq, [np.abs(dif/2), dif/T, -T/2, mean])
                tan = np.linalg.norm(seq - fitter_tan(x_data, *popt_tan))
            except:
                att += 1
                tan = 1e6
            try:
                popt_norm, pcov_norm = curve_fit(fitter_norm, x_data, seq, [np.abs(ampl), -1/2/T, T/2, np.min(seq)])
                norm = np.linalg.norm(seq - fitter_norm(x_data, *popt_norm))
            except:
                att +=1
                norm = 1e6
            if att == 2:
                sys.exit(f"Failed to fit {i}th electrode. Needs custom curve fitting")
            
            if tan > norm:
                funcs.append('((%5.6f) * exp((%5.6f) * (step*dt - (%5.6f))^2) + (%5.6f))' % tuple(popt_norm))
            else:
                funcs.append('((%5.6f) * (1 - 2/(exp(2*((%5.6f) * (step*dt + (%5.6f)))) + 1)) + (%5.6f))' % tuple(popt_tan))
                
    return funcs

def lossf_shuttle(uset, s, omegas, dots, L, dc_set, shuttlers, a):
    """
    Loss function for search of the optimal voltage sequence

    """
    rfss = np.array([0 for el in s if el.rf])
    for c, elec in enumerate(shuttlers):
        dc_set[elec] = uset[c]
    u_set = np.concatenate([rfss, np.array(dc_set)])
    loss = 0
    attempts = 0
    with s.with_voltages(dcs = u_set, rfs = None):
        for i,x in enumerate(dots):
            while attempts < 200:
                try:
                    try:
                        xreal = s.minimum(x + 1e-6/L*np.array([(-1)**attempts*attempts/10, 0, 1e-6]), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
                        break
                    except:
                        try:
                            xreal = s.minimum(x + 1e-6/L*np.array([1e-6, (-1)**attempts*attempts/10, 1e-6]), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
                            break
                        except:
                            xreal = s.minimum(x + 1e-6/L*np.array([1e-6, 1e-6, (-1)**attempts*attempts/10]), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
                            break
                except:
                    attempts += 1
                    xreal = x 

            curv_z, mod_dir=s.modes(xreal,sorted=False)
            loss += np.linalg.norm(xreal - x) + a*L**(-2)*(curv_z[0]-omegas[i])**2
    return loss


def shuttling_voltage(s, starts, routes, T, dc_set, shuttlers = 0, vmin = -15, vmax = 15, res = 50, L = 1e-6, need_func = False, a = 0):
    '''
    General function for shuttling voltage sequence, which optimizes the voltage
    sequence on chosen electrode for the (arbitrary 3D) route of 1 ion or 
    simulataneous varying routes of several different ions (potential wells)

    Parameters
    ----------
    s : electrode.System object
        Surface trap from electrode package
    starts : list shape (number of wells, 3)
        array of initial coordinates of each potential well to be considered
        may be only one point with shape (3)
    routes : function(i:int, t:float/np.array)
        function determining the routes of potential wells in time. Must be in
        the following form: if only 1 well:
            def routes(t):
                return np.array([x(t), y(t), z(t)])
        if 2 or more wells shuttled simultaneously:
            def routes(i,t):
                if i == 0:
                    return np.array([x_0(t), y_0(t), z_0(t)])
                ...
                if i == j:
                    return np.array([x_j(t), y_j(t), z_j(t)])          
    T : float
        time of shuttling operation
    dc_set : list shape(len(DCs))
        initial voltages on all dc electrodes
    shuttlers : list, optional, default is 0
        list of DC electrodes (defined as numbers), used for shuttling
        if ==0 then all dc electrodes participate in shuttling
        if ==[2,4,6] for example, only 2, 4 and 6th electrodes are participating,
        the rest are stationary dc
    vmin : float, optional
        Minimal allowable DC voltage. The default is -15.
    vmax : float, optional
        Maximal allowable DC voltage. The default is 15.
    res : int, optional
        Number of steps of the voltage sequence during shuttling time.
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.
    need_func : bool, optional
        if True, the approximation functions of voltage sequences on all DC electrodes
        are provided, which are used for MD simulation of shuttling
    a : float, optional
        Degree of considering secular frequency varience during the route.
        The default is 0, meaning no consideration. If it appears during analysis,
        that the optimized voltage sequence vary frequencies significantly, you
        can turn it one, but otherwise it just slows the optimization.

    Returns
    -------
    voltage_seq : list shape (len(shuttlers), res+1)
        voltage sequences on each shuttler electrode (could be constant)
    funcs : list of strings shape (len(DCs)), optional
        list of functions in format, used for MD simulation (including stationary DC)

    '''
    voltage_seq = []
    if shuttlers == 0:
        shuttlers = range(len(dc_set))
    try:
        starts[0][0]
    except:
        starts = np.array([starts])
    bnds = [(vmin,vmax) for el in shuttlers]
    rfss = np.array([0 for el in s if el.rf])
    u_set = np.concatenate([rfss, np.array(dc_set)])
    wells = len(starts)
    curves = np.zeros(wells)
    for i,start in enumerate(starts):
        with s.with_voltages(dcs = u_set, rfs = None):
            x1 = s.minimum(np.array(start), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
            curv_z, mod_dir=s.modes(x1,sorted=False)
            curves[i] = (curv_z[0])
    uset = np.zeros(len(shuttlers))
    for c, elec in enumerate(shuttlers):
        uset[c] = dc_set[elec]
    x = np.zeros((wells, 3))
    for dt in tqdm(range(res+1), desc='Shuttling optimization'):
        t = dt*T/res
        for i,x0 in enumerate(starts):
            try:
                x[i] = x0 + routes(i,t)
            except:
                x[i] = x0 + routes(t)
        newset = minimize(lossf_shuttle, uset, args = (s, curves, x, L, dc_set, shuttlers, a), tol = 1e-9, bounds = bnds, options = {'maxiter' : 1000000})
        uset = newset.x
        voltage_seq.append(uset) 
        
    voltage_seq = np.array(voltage_seq).T
    volt_seq = []
    k = 0
    for i, v in enumerate(dc_set):
        if i in shuttlers:
            volt_seq.append(voltage_seq[k])
            k += 1
        else:
            volt_seq.append(np.array([v]*(res+1)))
    volt_seq = np.array(volt_seq)
    
    if need_func:
        funcs = approx_linear_shuttling(volt_seq, T, dc_set, shuttlers, res)
        return volt_seq, funcs
    else:
        return volt_seq

@lammps.fix
def polygon_shuttling(uid, Omega, rf_set, RFs, DCs, shuttlers, cover=(0, 0)):
    """
    Simulates an arbitrary planar trap with polygonal electrodes
    and shuttling in this trap. The shuttling is specified by the 
    voltage sequence on the DC electrodes, in terms of some smooth function V(t).
    Any V(t) can be applied to this function. It must be specifyed as a string.
    For example: V(t) = np.exp(-k*t**2) == "exp(-k*(step*dt)^2)",
    If const: V(t) = V == "V"
    
    Parameters
    ----------
    uid : str
        ions ID, .self parameter
    Omega : float or list shape len(RFs)
        array of RF frequencies (or a frequency if they are the same) of each RF electrode
    rf_set : array shape (len(RFs))
        Set of the peak voltages on RF electrodes.
    RFs : list shape (len(RFs), 4)
        Array of coordinates of RF electrodes in m
    DCs : list shape (len(DCs), 4)
        Array of coordinates of DC electrodes in m
        The order of electrodes must match the order of voltage set and omegas.
    shuttlers : list of strings shape (len(DCs))
        list of strings, representing functions at which voltage on DC electrodes
        is applied through simulation time
    cover : list shape (2), optional
        array [cover_number, cover_height] - number of the terms in
        cover electrode influence expansion (5 is mostly enough) and its height.

    Returns
    -------
    odict : str
        updates simulation lines, which will be executed by LAMMPS
    """
    odict = {}
    rf = [len(a) for a in RFs]
    nrf = len(rf)
    try:
        Omega[0]
    except:
        Omega = np.ones(nrf)*Omega
    Omega = np.array(Omega)
    lines = [
        f'\n# Creating a polygonal Surface Electrode Trap... (fixID={uid})']
    odict['timestep'] = 1 / np.max(Omega) / 20
    for i in range(nrf):
        lines.append(
            f'variable phase{uid}{i:d}\t\tequal "{Omega[i]:e}*step*dt"')
    xcc = []
    ycc = []
    zcc = []

    for iterr in range(nrf):
        xc = []
        yc = []
        zc = []
        polygon = np.array(RFs[iterr])
        no = np.array(polygon).shape[0]

        for m in range(2*cover[0]+1):
            xt = f'(x - ({polygon[no-1, 0]:e}))'
            numt = no-1
            yt = f'(y - ({polygon[no-1, 1]:e}))'
            cov = 2*(m - cover[0])*cover[1]
            if cov == 0:
                z = 'z'
            else:
                z = f'(z + ({cov:e}))'

            for k in range(no):
                xo = xt
                yo = yt
                numo = numt
                numt = k
                dx = polygon[numt, 0] - polygon[numo, 0]
                dy = polygon[numo, 1] - polygon[numt, 1]
                c = polygon[numt, 0] * polygon[numo, 1] - polygon[numo, 0] * polygon[numt, 1]
                lt = (polygon[numt, 0] - polygon[numo, 0]) ** 2 + (polygon[numt, 1] - polygon[numo, 1]) ** 2
                ro = f'sqrt({xo}^2+{yo}^2+{z}^2)'
                xt = f'(x - ({polygon[k, 0]:e}))'
                yt = f'(y - ({polygon[k, 1]:e}))'
                rt = f'sqrt({xt}^2+{yt}^2+{z}^2)'
                
                n = f'({ro}+{rt})/({ro}*{rt}*(({ro}+{rt})^2-({lt:e})))'
                if dx != 0:
                    yc.append(
                        f'({dx:e})*{z}*{n}')
                if dy != 0:
                    xc.append(
                        f'({dy:e})*{z}*{n}')
                if (c**2 + dx**2 + dy**2 == 0):
                    pass
                elif (c**2 + dx**2 == 0):
                    zc.append(f'(-({dy:e})*x)*{n}')
                elif (c**2 + dy**2 == 0):
                    zc.append(f'(-({dx:e})*y)*{n}')
                elif (dx**2 + dy**2 == 0):
                    zc.append(f'({c:e})*{n}')
                elif (c == 0):
                    zc.append(f'(- ({dx:e})*y - ({dy:e})*x)*{n}')
                elif (dx == 0):
                    zc.append(f'({c:e} - ({dy:e})*x)*{n}')
                elif (dy == 0):
                    zc.append(f'({c:e} - ({dx:e})*y)*{n}')
                else:
                    zc.append(f'({c:e} - ({dx:e})*y - ({dy:e})*x)*{n}')
        xc = ' + '.join(xc)
        yc = ' + '.join(yc)
        zc = ' + '.join(zc)
        
        xcc.append(f'({rf_set[iterr]/np.pi:e})*({xc})*cos(v_phase{uid}{iterr:d})')
        ycc.append(f'({rf_set[iterr]/np.pi:e})*({yc})*cos(v_phase{uid}{iterr:d})')
        zcc.append(f'({rf_set[iterr]/np.pi:e})*({zc})*cos(v_phase{uid}{iterr:d})')
        
    xcc = ' + '.join(xcc)
    ycc = ' + '.join(ycc)
    zcc = ' + '.join(zcc)

    xrr = []
    yrr = []
    zrr = []
    for ite, elem in enumerate(DCs):
        xr = []
        yr = []
        zr = []
        polygon = np.array(elem)
        no = np.array(polygon).shape[0]

        for m in range(2 * cover[0] + 1):
            x2 = f'(x - ({polygon[no-1, 0]:e}))'
            y2 = f'(y - ({polygon[no-1, 1]:e}))'
            numt = no - 1
            cov = 2 * (m - cover[0]) * cover[1]
            if cov == 0:
                z = 'z'
            else:
                z = f'(z + ({cov:e}))'

            for k in range(no):
                numo = numt
                numt = k
                rodc = f'sqrt({x2}^2+{y2}^2+{z}^2)'
                x2 = f'(x - ({polygon[k, 0]:e}))'
                y2 = f'(y - ({polygon[k, 1]:e}))'
                rtdc = f'sqrt({x2}^2+{y2}^2+{z}^2)'
               
                dx = polygon[numt, 0] - polygon[numo, 0]
                dy = polygon[numo, 1] - polygon[numt, 1]
                c = polygon[numt, 0] * polygon[numo, 1] - polygon[numo, 0] * polygon[numt, 1]
                lt = (polygon[numt, 0] - polygon[numo, 0]) ** 2 + (polygon[numt, 1] - polygon[numo, 1]) ** 2
                n = f'({rodc}+{rtdc})/({rodc}*{rtdc}*(({rodc}+{rtdc})^2-({lt:e})))'

                if dx != 0:
                    yr.append(
                        f'({dx:e})*{z}*{n}')
                if dy != 0:
                    xr.append(
                        f'({dy:e})*{z}*{n}')
                if (c**2 + dx**2 + dy**2 == 0):
                    pass
                elif (c**2 + dx**2 == 0):
                    zr.append(f'(-({dy:e})*x)*{n}')
                elif (c**2 + dy**2 == 0):
                    zr.append(f'(-({dx:e})*y)*{n}')
                elif (dx**2 + dy**2 == 0):
                    zr.append(f'({c:e})*{n}')
                elif (c == 0):
                    zr.append(f'(- ({dx:e})*y - ({dy:e})*x)*{n}')
                elif (dx == 0):
                    zr.append(f'({c:e} - ({dy:e})*x)*{n}')
                elif (dy == 0):
                    zr.append(f'({c:e} - ({dx:e})*y)*{n}')
                else:
                    zr.append(f'({c:e} - ({dx:e})*y - ({dy:e})*x)*{n}')
        
        xr = ' + '.join(xr)
        yr = ' + '.join(yr)
        zr = ' + '.join(zr)
        
        xrr.append(f'({-1/np.pi:e})*({xr})*{shuttlers[ite]}')
        yrr.append(f'({-1/np.pi:e})*({yr})*{shuttlers[ite]}')
        zrr.append(f'({-1/np.pi:e})*({zr})*{shuttlers[ite]}')
    xrr = ' + '.join(xrr)
    yrr = ' + '.join(yrr)
    zrr = ' + '.join(zrr)
    lines.append(f'variable oscEX{uid} atom "{xcc}+{xrr}"')
    lines.append(f'variable oscEY{uid} atom "{ycc}+{yrr}"')
    lines.append(f'variable oscEZ{uid} atom "{zcc}+{zrr}"')
    lines.append(
        f'fix {uid} all efield v_oscEX{uid} v_oscEY{uid} v_oscEZ{uid}\n')
 
    odict.update({'code': lines})

    return odict



"""
Design functions for traps.
"""

def circle_packaging(scale, boundary, n, res):
    """
    Function to almost optimally approximate the given shape (specified by boundary)
    with circles for a given resolution. Works as follows:
        the area of scale*scale size is packed with circles by hexagonal
        packing with resolution res. Then, for nth electrode the shape, specified
        by boundary() is cutted from the area.

    Parameters
    ----------
    scale : float
        size of packed area
    boundary : function(n:int, x:list of in plane coordinates [x,y])
        boundary function for a given electrode's shape. Returns True, is the 
        point is within boundaries, or False, if not.
        Example for 2 circle electrodes:
        def boundaries(i, x):
            if i == 0:
                if (x[0]**2+x[1]**2<1):
                    return True  
                else:
                    return False
            if i == 1:
                if ((x[0]-2)**2+(x[1]-2)**2<1):
                    return True  
                else:
                    return False
    n : int
        index of an electrode
    res : int
        resolution

    Returns
    -------
    points : list shape (number of points, 2)
        coordinates of centers, falling into the boundaries
    areas : list shape (number of points)
        areas of each point (equal float)

    """
    centers = [[[i + j*.5, j*3**.5*.5] for j in range(-res - min(0, i), res - max(0, i) + 1)] for i in range(-res, res + 1)]
    x = np.vstack(centers)/(res + .5)*scale # centers
    
    a = np.ones(len(x))*3**.5/(res + .5)**2/2*scale**2 # areas
    
    ps = []
    ars = []
    
    for i, el in enumerate(x):
        if boundary(n, el):
            ps.append(el)
            ars.append(a[i])
    return np.array(ps), np.array(ars)


def five_wire_trap_design(Urf, DCtop, DCbottom, cwidth, clength, boardwidth, rftop, rflength, rfbottom, need_coordinates = False, L = 1e-6, patternTop = 1, patternBot = 1, getCoordinate = False, gapped = 0, cheight=0, cmax=0, need_plot = False, save_plot = None):
    '''
    Elaborate function, designing arbitrary five-wire trap.

    Parameters
    ----------
    Urf : float
        Scaled RF amplitude of the trap. If scaled from SU rf amplitude (Vrf) 
        as Urf = Vrf*np.sqrt(Z/mass)/(2*L*Omega), where Z - ion's charge in SU,
        then accounts for pseudopotential approximation.
    DCtop : list shape (number of electrodes, 2)
        list of [length (along y-axis), width(along x-axis)] of electrodes on top of the RF line. 
        Placed in the middle of the rf line.
    DCbottom : list shape (number of electrodes, 2)
        list of [length (along y-axis), width(along x-axis)] of electrodes bottom of the RF line. 
        Placed in the middle of the rf line.
    cwidth : float
        Width of the central dc electrode (along y-axis).
    clength : float
        Length of the central dc electrode (along x-axis).
    boardwidth : float
        Width of the gap between electrodes. Will be filled equally from both electrodes, 
        sharing this gap, as per House, Analytical model for surface traps.
    rftop : float
        Width of the upper rf electrode in the rf line (along y-axis).
    rflength : float
        Length of the rf line (along x-axis). If >clength, it will encapsulate the central dc electrode
    rfbottom : float
        Width of the lower rf electrode in the rf line (along y-axis).
    need_coordinates : bool, optional
        If True, returns the coordinates, scaled by L to SU form, used by polygon_simulation(). The default is False.
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.
    patternTop : int, optional
        It will repeat the DCtop electrodes patternTop times. The default is 1.
    patternBot : int, optional
        It will repeat the DCbottom electrodes patternBot times. The default is 1.
    getCoordinate : bool, optional
        If True, it will create a file with coordinates, considering the gaps, 
        for GDS file production. The default is False.
    gapped : float, optional
        Creates gaps between central dc electrodes and RF line with given width.
        Should be used only if said gap is very large, so capacitive coupling will not affect the simulation.
        The default is 0.
    cheight : float, optional
        Height of cover electrode - grounded plane above the trap. The default is 0.
    cmax : int, optional
        Expansion order for cover electrode. Usually 5 is sufficient for correct calculations. The default is 0.
    need_plot : bool, optional
        If True, returns a plot of the trap with specified RF electrode. The default is False.
    save_plot : str, optional
        Saves the plot to the file for the provided path.
        For example: "five-wire_trap.eps" will create this file with the plot.
        The default is None.

    Returns
    -------
    s : electrode.System object
        Surface trap from electrode .
    RF_electrodes : list shape (number of RF electrodes, electrode shape), optional
        returns RF electrodes coordinates in SU for MD simulation.
    DC_electrodes : list shape (number of DC electrodes, electrode shape), optional
        returns DC electrodes coordinates in SU for MD simulation.
    '''
    clefttop = [-clength / 2, cwidth]

    c = [[clefttop[0] - boardwidth / 2, clefttop[1] + boardwidth / 2],
         [clefttop[0] + clength + boardwidth, clefttop[1] + boardwidth / 2],
         [clefttop[0] + clength + boardwidth, -boardwidth / 2], [clefttop[0] - boardwidth / 2, -boardwidth / 2]]
    rf_top = [[-rflength / 2, rftop + boardwidth + gapped + clefttop[1]],
              [rflength / 2, rftop + boardwidth + gapped + clefttop[1]],
              [rflength / 2, clefttop[1] + boardwidth / 2 + gapped],
              [-rflength / 2, clefttop[1] + boardwidth / 2 + gapped]]
    rf_bottom = [[-rflength / 2, boardwidth / 2 + clefttop[1] + gapped],
                 [clefttop[0] - boardwidth / 2,
                     clefttop[1] + boardwidth / 2 + gapped],
                 [clefttop[0] - boardwidth / 2, -boardwidth / 2 - gapped],
                 [clefttop[0] + boardwidth + clength, -boardwidth / 2 - gapped],
                 [clefttop[0] + boardwidth + clength,
                     clefttop[1] + boardwidth / 2 + gapped],
                 [rflength / 2, clefttop[1] + boardwidth / 2 +
                     gapped], [rflength / 2, -rfbottom - boardwidth - gapped],
                 [-rflength / 2, -rfbottom - boardwidth - gapped]]
    bd = boardwidth
    DCb = DCbottom

    # define arrays of DCs considering pattern
    DCtop = DCtop * patternTop
    DCtop = np.array(DCtop)

    DCb = DCb * patternBot
    DCb = np.array(DCb)

    # Part, defining top DC electrods
    n = len(DCtop)
    # define m -- number of the DC, which will start in x=0
    m = n // 2

    gapped = 2 * gapped

    # define t -- 3d array containing 2d arrays of top DC electrodes
    t = [[[0 for i in range(2)] for j in range(4)] for k in range(n)]
    t = np.array(t)

    # starting point - central electrode, among top DCs. Considering parity of electrode number (i just want it to be BEAUTIFUL)
    if n % 2 == 0:
        t[m] = np.array([[-bd / 2, rftop + bd * 3 / 2 + clefttop[1] + DCtop[m][0] + gapped],
                         [bd / 2 + DCtop[m][1], rftop + bd * 3 /
                             2 + clefttop[1] + DCtop[m][0] + gapped],
                         [bd / 2 + DCtop[m][1], rftop + bd + clefttop[1] + gapped],
                         [-bd / 2, rftop + bd + clefttop[1] + gapped]])
    else:
        t[m] = np.array([[-bd / 2 - DCtop[m][1] / 2, rftop + bd * 3 / 2 + clefttop[1] + DCtop[m][0] + gapped],
                         [bd / 2 + DCtop[m][1] / 2, rftop + bd * 3 /
                             2 + clefttop[1] + DCtop[m][0] + gapped],
                         [bd / 2 + DCtop[m][1] / 2, rftop +
                             bd + clefttop[1] + gapped],
                         [-bd / 2 - DCtop[m][1] / 2, rftop + bd + clefttop[1] + gapped]])

    # adding electrodes to the right of central DC
    for k in range(m, n - 1):
        t[k + 1] = np.array([t[k][1] + np.array([0, DCtop[k + 1][0] - DCtop[k][0]]),
                             t[k][1] + np.array([bd + DCtop[k + 1]
                                                [1], DCtop[k + 1][0] - DCtop[k][0]]),
                             t[k][2] + np.array([bd + DCtop[k + 1][1], 0]), t[k][2]])

    # adding electrodes to the left
    for k in range(1, m + 1):
        r = m - k
        t[r] = np.array([t[r + 1][0] + np.array([-bd - DCtop[r][1], DCtop[r][0] - DCtop[r + 1][0]]),
                         t[r + 1][0] +
                         np.array([0, DCtop[r][0] - DCtop[r + 1][0]]),
                         t[r + 1][3], t[r + 1][3] + np.array([-bd - DCtop[r][1], 0])])

    # Part for bottom DCs
    nb = len(DCb)
    m = nb // 2
    b = [[[0 for i in range(2)] for j in range(4)] for k in range(nb)]
    b = np.array(b)

    # starting electrode
    if nb % 2 == 0:
        b[m] = np.array(
            [[-bd / 2, -rfbottom - boardwidth - gapped], [bd / 2 + DCb[m][1], -rfbottom - boardwidth - gapped],
             [bd / 2 + DCb[m][1], -rfbottom - bd * 3 / 2 - DCb[m][0] - gapped],
             [-bd / 2, -rfbottom - bd * 3 / 2 - DCb[m][0] - gapped]])
    else:
        b[m] = np.array([[-bd / 2 - DCb[m][1] / 2, -rfbottom - boardwidth - gapped],
                         [bd / 2 + DCb[m][1] / 2, -rfbottom - boardwidth - gapped],
                         [bd / 2 + DCb[m][1] / 2, -rfbottom -
                             bd * 3 / 2 - DCb[m][0] - gapped],
                         [-bd / 2 - DCb[m][1] / 2, -rfbottom - bd * 3 / 2 - DCb[m][0] - gapped]])

    # Adding DCs. The same algorythm exept sign y-ax
    for k in range(m, nb - 1):
        b[k + 1] = np.array([b[k][1] + np.array([0, 0]), b[k][1] + np.array([bd + DCb[k + 1][1], 0]),
                             b[k][2] + np.array([bd + DCb[k + 1]
                                                [1], -DCb[k + 1][0] + DCb[k][0]]),
                             b[k][2] + np.array([0, -DCb[k + 1][0] + DCb[k][0]])])

    for k in range(1, m + 1):
        r = m - k
        b[r] = np.array([b[r + 1][0] + np.array([-bd - DCb[r][1], 0]), b[r + 1][0] + np.array([0, 0]),
                         b[r + 1][3] +
                         np.array([0, -DCb[r][0] + DCb[r + 1][0]]),
                         b[r + 1][3] + np.array([-bd - DCb[r][1], -DCb[r][0] + DCb[r + 1][0]])])

    # Reverse every electrode for correct calculation
    rf_top = rf_top[::-1]
    rf_bottom = rf_bottom[::-1]
    c = c[::-1]
    for i in range(n):
        t[i] = t[i][::-1]
    for i in range(nb):
        b[i] = b[i][::-1]

    # Creating array of electrodes with names
    electrodes = [
        (" ", [rf_top,
                rf_bottom])]
    for i in range(n):
        st = "t[" + str(i + 1) + "]"
        electrodes.append([st, [t[i]]])
    for i in range(nb):
        st = "b[" + str(i + 1) + "]"
        electrodes.append([st, [b[i]]])
    electrodes.append(["c", [c]])

    # Polygon approach. All DCs are 0 for nown
    s = System([PolygonPixelElectrode(cover_height=cheight, cover_nmax=cmax, name=n, paths=map(np.array, p))
                for n, p in electrodes])
    s[" "].rf = Urf

    for i in range(n):
        st = "t[" + str(i + 1) + "]"
        s[st].dc = 0
    for i in range(nb):
        st = "b[" + str(i + 1) + "]"
        s[st].dc = 0
    s["c"].dc = 0

    # Exact coordinates for lion
    elec = [
        ("", [np.array(rf_top) * L,
                np.array(rf_bottom) * L])]
    for i in range(n):
        st = "t[" + str(i + 1) + "]"
        elec.append([st, [np.array(t[i]) * L]])
    for i in range(nb):
        st = "b[" + str(i + 1) + "]"
        elec.append([st, [np.array(b[i]) * L]])
    elec.append(["c", [np.array(c) * L]])

    # Part of getting coordinates of electrodes
    if getCoordinate:
        # once again to add gaps properly, lol
        rf_top = rf_top[::-1]
        rf_bottom = rf_bottom[::-1]
        c = c[::-1]
        for i in range(n):
            t[i] = t[i][::-1]
        for i in range(nb):
            b[i] = b[i][::-1]

        for i in range(n):
            # moves to the left on the boardwidth/2
            t[i][0][0] = t[i][0][0] + bd / 2
            t[i][1][0] = t[i][1][0] - bd / 2  # to the right
            t[i][2][0] = t[i][2][0] - bd / 2
            # moves up as C-point of electrode
            t[i][2][1] = t[i][2][1] + bd / 2
            t[i][3][0] = t[i][3][0] + bd / 2
            t[i][3][1] = t[i][3][1] + bd / 2
        for i in range(nb):
            # moves to the left on the boardwidth/2
            b[i][0][0] = b[i][0][0] + bd / 2
            b[i][1][0] = b[i][1][0] - bd / 2  # to the right
            b[i][2][0] = b[i][2][0] - bd / 2
            b[i][0][1] = b[i][0][1] - bd / 2
            b[i][3][0] = b[i][3][0] + bd / 2
            b[i][1][1] = b[i][1][1] - bd / 2
        c[0][0] = c[0][0] + bd / 2
        c[0][1] = c[0][1] - bd / 2
        c[1][0] = c[1][0] - bd / 2
        c[1][1] = c[1][1] - bd / 2
        c[2][0] = c[2][0] - bd / 2
        c[2][1] = c[2][1] + bd / 2
        c[3][0] = c[3][0] + bd / 2
        c[3][1] = c[3][1] + bd / 2
        rf_top[0][1] = rf_top[0][1] - bd / 2
        rf_top[1][1] = rf_top[1][1] - bd / 2
        rf_top[2][1] = rf_top[2][1] + bd / 2
        rf_top[3][1] = rf_top[3][1] + bd / 2
        # for rf_top and rf_bottom to be connected
        rf_bottom[0][1] = rf_bottom[0][1] + bd / 2
        rf_bottom[1][1] = rf_bottom[1][1] + bd / 2
        rf_bottom[1][0] = rf_bottom[1][0] - bd / 2
        rf_bottom[2][1] = rf_bottom[2][1] - bd / 2
        rf_bottom[2][0] = rf_bottom[2][0] - bd / 2
        rf_bottom[3][0] = rf_bottom[3][0] + bd / 2
        rf_bottom[3][1] = rf_bottom[3][1] - bd / 2
        rf_bottom[4][1] = rf_bottom[4][1] + bd / 2
        rf_bottom[4][0] = rf_bottom[4][0] + bd / 2
        rf_bottom[5][1] = rf_bottom[5][1] + bd / 2
        rf_bottom[6][1] = rf_bottom[6][1] + bd / 2
        rf_bottom[7][1] = rf_bottom[7][1] + bd / 2

        # array of coordinates with names
        coordinates = [
            ("RF", [rf_top,
                    rf_bottom])]
        for i in range(n):
            st = "t[" + str(i + 1) + "]"
            coordinates.append([st, [t[i]]])
        for i in range(nb):
            st = "b[" + str(i + 1) + "]"
            coordinates.append([st, [b[i]]])
        coordinates.append(["c", [c]])

        # Writes them in a file
        with open('coordinates.txt', 'w') as f:
            for item in coordinates:
                f.write(f'{item}\n')

    # creates a plot of electrode
    if need_plot:
        fig, ax = plt.subplots(1, 2, figsize=(60, 60))
        s.plot(ax[0])
        s.plot_voltages(ax[1], u=s.rfs)
        xmax = rflength * 2 / 3
        ymaxp = (np.max(DCtop) + rftop + clefttop[1]) * 1.2
        ymaxn = (np.max(DCbottom) + rfbottom) * 1.2
        ax[0].set_title("colour")
        ax[1].set_title("rf-voltages")
        for axi in ax.flat:
            axi.set_aspect("equal")
            axi.set_xlim(-xmax, xmax)
            axi.set_ylim(-ymaxn, ymaxp)
        if save_plot:
            plt.tight_layout()
            plt.savefig(save_plot)
    if need_coordinates:
        RF_electrodes=[]
        for iterr in range(2):
            RF_electrodes.append(elec[0][1][iterr])
        DC_electrodes = []
        for ite in range(n + nb + 1):
            DC_electrodes.append(elec[ite + 1][1][0])
        return s, RF_electrodes, DC_electrodes

    else:
        return s


def ring_trap_design(Urf, Omega, r, R, r_dc = 0, v_dc = 0, res = 100, need_coordinates = False, need_plot = False, cheight=0, cmax=0, save_plot = None):
    '''
    Function for designing the ring RF trap with (optionally) central DC electrode

    Parameters
    ----------
    Urf : float
        Scaled RF amplitude of the trap. If scaled from SU rf amplitude (Vrf) 
        as Urf = Vrf*np.sqrt(Z/mass)/(2*L*Omega), where Z - ion's charge in SU,
        then accounts for pseudopotential approximation.
    Omega : float
        RF frequency of the trap.
    r : float
        inner radius of the electrode ring in SU.
    R : float
        outter radius of the electrode ring in SU.
    r_dc : float, optional
        radius of central DC circled electrode. The default is 0.
    v_dc : float, optional
        Voltage of dc electrode. The default is 0.
    res : int, optional
        Resolution of the trap. The default is 100.
    need_coordinates : bool, optional
        If True, returns "trap" object for point_trap() simulation. The default is False.
    need_plot : bool, optional
        If True, returns a plot of the trap, with RF and DC electrode voltages separately. The default is False.
    cheight : float, optional
        Height of cover electrode - grounded plane above the trap. The default is 0.
    cmax : int, optional
        Expansion order for cover electrode. Usually 5 is sufficient for correct calculations. The default is 0.
    save_plot : str, optional
        Saves the plot to the file for the provided path.
        For example: "five-wire_trap.eps" will create this file with the plot.
        The default is None.

    Returns
    -------
    s : electrode.System object
        Surface trap from electrode.
    trap : list, optional
        List of two elements: [trap_rf, trap_dc], for simulation with point_trap()
        trap_rf: lits of rf electrodes, each contains 4 parameters: 
            [frequency: RF frequency of the electrode,
             voltage: peak voltage of the electrode, 
             centers: coordinates of points' centers, 
             areas: areas of each point]
        trap_dc: lits of dc electrodes (can be empty), each contains 4 parameters: 
            [voltage: voltage of the electrode,
             centers: coordinates of points' centers, 
             areas: areas of each point]
    '''
    def ring_boundary(i, x):
        if i == 0:
            if (x[0]**2+x[1]**2 > r**2) and (x[0]**2+x[1]**2 < R**2):
                return True
            else:
                return False
        elif i == 1:
            if (x[0]**2+x[1]**2 < r_dc**2):
                return True
            else:
                return False
            
    frequencies = [Omega] 
    rf_voltages = [Urf]
    dc_voltages = [v_dc]
    scale = R*1.5
    
    return point_trap_design(frequencies, rf_voltages, dc_voltages, ring_boundary, scale, res, need_coordinates = need_coordinates, need_plot = need_plot, cheight=cheight, cmax=cmax, save_plot = save_plot)

def point_trap_design(frequencies, rf_voltages, dc_voltages, boundaries, scale, resolution, need_coordinates = False, need_plot = False, cheight=0, cmax=0, save_plot = None):
    '''
    Function for designing arbitrarily shape point trap.
    Each electrode shape is determined by provided boundary.

    Parameters
    ----------
    frequencies : float or list shape len(RFs)
        array of RF frequencies (or a frequency if they are the same) of each RF electrode
    rf_voltages : array shape (len(RFs))
        Set of the peak voltages on RF electrodes.
    dc_voltages : array shape (len(DCs))
        Set of the voltages on DC electrodes.
    boundary : function(n:int, x:list of in plane coordinates [x,y])
        boundary function for a given electrode's shape. Returns True, is the 
        point is within boundaries, or False, if not.
        Example for 2 circle electrodes:
        def boundaries(i, x):
            if i == 0:
                if (x[0]**2+x[1]**2<1):
                    return True  
                else:
                    return False
            if i == 1:
                if ((x[0]-2)**2+(x[1]-2)**2<1):
                    return True  
                else:
                    return False
    scale : float
        size of the trap will be scale*scale.
    resolution : int, optional
        Resolution of the trap. The default is 100.
    need_coordinates : bool, optional
        If True, returns "trap" object for point_trap() simulation. The default is False.
    need_plot : bool, optional
        If True, returns a plot of the trap, with RF and DC electrode voltages separately. The default is False.
    cheight : float, optional
        Height of cover electrode - grounded plane above the trap. The default is 0.
    cmax : int, optional
        Expansion order for cover electrode. Usually 5 is sufficient for correct calculations. The default is 0.
    save_plot : str, optional
        Saves the plot to the file for the provided path.
        For example: "five-wire_trap.eps" will create this file with the plot.
        The default is None.

    Returns
    -------
    s : electrode.System object
        Surface trap from electrode.
    trap : list, optional
        List of two elements: [trap_rf, trap_dc], for simulation with point_trap()
        trap_rf: lits of rf electrodes, each contains 4 parameters: 
            [frequency: RF frequency of the electrode,
             voltage: peak voltage of the electrode, 
             centers: coordinates of points' centers, 
             areas: areas of each point]
        trap_dc: lits of dc electrodes (can be empty), each contains 4 parameters: 
            [voltage: voltage of the electrode,
             centers: coordinates of points' centers, 
             areas: areas of each point]

    '''
    x = []
    a = []
    x_rf = []
    x_dc = []
    a_rf = []
    a_dc = []
    voltages_rf = []
    voltages_dc = []
    trap_rf = []
    trap_dc = []
    omegas = []
    n_rf = 0
    n_dc = 0
    k = len(rf_voltages)
    try:
        frequencies[0]
    except:
        frequencies = np.ones(k)*frequencies
    frequencies = np.array(frequencies)
    for i, voltage in enumerate(rf_voltages):
        centers, areas = circle_packaging(scale, boundaries, i, resolution)
        x_rf.append(centers)
        a_rf.append(areas)
        n = len(areas)
        voltages_rf.append(np.ones(n)*voltage)
        trap_rf.append([frequencies[i], voltage, areas, centers])
        omegas.append(np.ones(n)*frequencies[i])
        n_rf += n
        
    for i, voltage in enumerate(dc_voltages):
        centers, areas = circle_packaging(scale, boundaries, i+k, resolution)
        x_dc.append(centers)
        a_dc.append(areas)
        trap_dc.append([voltage, areas, centers])
        n = len(areas)
        voltages_dc.append(np.ones(n)*voltage)
        n_dc += n
        
    x_rf = np.concatenate(x_rf)
    a_rf = np.concatenate(a_rf)
    voltages_rf = np.concatenate(voltages_rf)
    omegas = np.concatenate(omegas)
    if len(dc_voltages) > 0:
        x_dc = np.concatenate(x_dc)
        a_dc = np.concatenate(a_dc)
        x = np.concatenate((x_rf, x_dc))
        a = np.concatenate((a_rf, a_dc))
        voltages_dc = np.concatenate(voltages_dc)
        rf = np.concatenate((voltages_rf, np.zeros(n_dc)))
        dc = np.concatenate((np.zeros(n_rf), voltages_dc))
    else:
        x = x_rf
        a = a_rf
        rf = voltages_rf
        dc = voltages_dc
         
    pointelectrode = [PointPixelElectrode(cover_height=cheight, cover_nmax=cmax, points=[xi], areas=[ai]) for
                    xi, ai in zip(x,a)] 
    s = System(pointelectrode)
    s.rfs = rf
    s.dcs = dc
    
    trap = [trap_rf, trap_dc]
    
    if need_plot:
        if len(dc_voltages)>0:
            fig, ax = plt.subplots(1,2,figsize=(13, 5))
            s.plot_voltages(ax[0], u=s.rfs)
            ax[0].set_xlim((-scale, scale))
            ax[0].set_ylim((-scale, scale))
            ax[0].set_title("RF voltage")
            s.plot_voltages(ax[1], u=s.dcs)
            ax[1].set_title('dc voltage')
            ax[1].set_xlim((-scale, scale))
            ax[1].set_ylim((-scale, scale))
            ax[0].set_aspect('equal', adjustable='box')
            ax[1].set_aspect('equal', adjustable='box')
            try:
                cmap = plt.cm.RdBu_r
                norm = mpl.colors.Normalize(vmin=np.min(dc_voltages), vmax=np.max(dc_voltages))
    
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, shrink=1, aspect=25)
    
                cb.ax.tick_params(labelsize=8)
                cb.set_label('Voltage, V', fontsize = 8)
            except:
                pass
            if save_plot:
                plt.savefig(save_plot, bbox_inches='tight')
            
        else:
            fig, ax = plt.subplots(1,2,figsize=(11.5, 5))
            s.plot_voltages(ax[0], u=s.rfs)
            ax[0].set_xlim((-scale, scale))
            ax[0].set_ylim((-scale, scale))
            ax[0].set_title("RF voltage")
            s.plot_voltages(ax[1], u=s.dcs)
            ax[1].set_title('dc voltage')
            ax[1].set_xlim((-scale, scale))
            ax[1].set_ylim((-scale, scale))
            ax[0].set_aspect('equal', adjustable='box')
            ax[1].set_aspect('equal', adjustable='box')
            if save_plot:
                plt.savefig(save_plot)
        plt.show()

    if need_coordinates:
        return s, trap
    else:
        return s


def n_rf_trap_design(Urf, DCtop, DCbottom, cwidth, rfwidth, rflength, n_rf=1, L = 1e-6, patternTop=1,
              patternBot=1, cheight=0, cmax=0, need_coordinates = False, need_plot = False, save_plot = None):
    '''
    Function for designing the surface trap, similar to five-wire trap, 
    but with n_rf RF electrodes on each side of the central one.

    Parameters
    ----------
    Urf : float
        Scaled RF amplitude of the trap. If scaled from SU rf amplitude (Vrf) 
        as Urf = Vrf*np.sqrt(Z/mass)/(2*L*Omega), where Z - ion's charge in SU,
        then accounts for pseudopotential approximation.
    DCtop : list shape (number of electrodes, 2)
        list of [length (along y-axis), width(along x-axis)] of electrodes on top of the RF line. 
        Placed in the middle of the rf line.
    DCbottom : list shape (number of electrodes, 2)
        list of [length (along y-axis), width(along x-axis)] of electrodes bottom of the RF line. 
        Placed in the middle of the rf line.
    cwidth : float
        Width of the central dc electrode (along y-axis).
    rfwidth : list of shape (n_rf, 2)
        List of widths of each rf electrode in the rf line (along y-axis),
        starting from closer to central electrodes, like [[width_upper, width_lower],...]
    rflength : float
        Length of the rf line (along x-axis). Equal to clength.
    n_rf : int, optional
            Number of rf electrodes on each side of the central dc electrode. 
            The default is 1, where it is just a five-wire trap
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.
    patternTop : int, optional
        It will repeat the DCtop electrodes patternTop times. The default is 1.
    patternBot : int, optional
        It will repeat the DCbottom electrodes patternBot times. The default is 1.
    cheight : float, optional
        Height of cover electrode - grounded plane above the trap. The default is 0.
    cmax : int, optional
        Expansion order for cover electrode. Usually 5 is sufficient for correct calculations. The default is 0.
    need_coordinates : bool, optional
        If True, returns the coordinates, scaled by L to SU form, used by polygon_simulation(). The default is False.
    need_plot : bool, optional
        If True, returns a plot of the trap with specified RF electrode. The default is False.
    save_plot : str, optional
        Saves the plot to the file for the provided path.
        For example: "five-wire_trap.eps" will create this file with the plot.
        The default is None.

    Returns
    -------
    s : electrode.System object
        Surface trap from electrode .
    RF_electrodes : list shape (number of RF electrodes, electrode shape), optional
        returns RF electrodes coordinates in SU for MD simulation.
    DC_electrodes : list shape (number of DC electrodes, electrode shape), optional
        returns DC electrodes coordinates in SU for MD simulation.

    '''
    c = [[-rflength/2, 0], [rflength/2, 0],
         [rflength/2, cwidth], [-rflength/2, cwidth]]
    RF = []
    electrodes = []
    for i in range(n_rf):
        tops = 0
        bots = 0
        for k in range(i):
            tops += rfwidth[k][0]
            bots += rfwidth[k][1]
        rf_top = [[-rflength/2, cwidth+tops], [rflength/2, cwidth+tops],
                  [rflength/2, cwidth+tops+rfwidth[i][0]], [-rflength/2, cwidth+tops+rfwidth[i][0]]]
        rf_bottom = [[-rflength / 2,  -bots-rfwidth[i][1]], [rflength / 2, -bots-rfwidth[i][1]],
                     [rflength / 2, -bots], [-rflength / 2, -bots]]
        st = 'rf' + str(i)+'u'
        electrodes.append([st, [rf_top]])
        st = 'rf' + str(i)+'d'
        electrodes.append([st, [rf_bottom]])
        RF.append(np.array(rf_top)*L)
        RF.append(np.array(rf_bottom) * L)
    # define arrays of DCs considering pattern
    pt = patternTop
    DCtop = DCtop * pt
    DCtop = np.array(DCtop)
    pb = patternBot
    DCb = DCbottom
    DCb = DCb * pb
    DCb = np.array(DCb)

    # Part, defining top DC electrods
    n = len(DCtop)
    # define m -- number of the DC, which will start in x=0
    m = n // 2

    # define t -- 3d array containing 2d arrays of top DC electrodes
    t = [[[0 for i in range(2)] for j in range(4)] for k in range(n)]
    t = np.array(t)
    lenrf_top = 0
    lenrf_bot = 0
    for i in range(n_rf):
        lenrf_top += rfwidth[i][0]
        lenrf_bot += rfwidth[i][1]
    
    DC = []
    if n % 2 == 0:
        t[m] = np.array([[0, lenrf_top+cwidth + DCtop[m][0]],
                         [DCtop[m][1], lenrf_top+cwidth + DCtop[m][0]],
                         [DCtop[m][1], lenrf_top+cwidth],
                         [0, lenrf_top+cwidth]])
    else:
        t[m] = np.array([[- DCtop[m][1] / 2, lenrf_top+cwidth + DCtop[m][0]],
                         [+ DCtop[m][1] / 2, lenrf_top+cwidth + DCtop[m][0]],
                         [+ DCtop[m][1] / 2, lenrf_top+cwidth],
                         [- DCtop[m][1] / 2, lenrf_top+cwidth]])

    # adding electrodes to the right of central DC
    for k in range(m, n - 1):
        t[k + 1] = np.array([t[k][1] + np.array([0, DCtop[k + 1][0] - DCtop[k][0]]),
                             t[k][1] + np.array([DCtop[k + 1][1],
                                                DCtop[k + 1][0] - DCtop[k][0]]),
                             t[k][2] + np.array([DCtop[k + 1][1], 0]), t[k][2]])

    # adding electrodes to the left
    for k in range(1, m + 1):
        r = m - k
        t[r] = np.array([t[r + 1][0] + np.array([- DCtop[r][1], DCtop[r][0] - DCtop[r + 1][0]]),
                         t[r + 1][0] +
                         np.array([0, DCtop[r][0] - DCtop[r + 1][0]]),
                         t[r + 1][3], t[r + 1][3] + np.array([- DCtop[r][1], 0])])

    # Part for bottom DCs
    nb = len(DCb)
    m = nb // 2
    b = [[[0 for i in range(2)] for j in range(4)] for k in range(nb)]
    b = np.array(b)

    # starting electrode
    if nb % 2 == 0:
        b[m] = np.array(
            [[0, -lenrf_bot], [+ DCb[m][1], -lenrf_bot],
             [+ DCb[m][1], -lenrf_bot - DCb[m][0]],
             [0, -lenrf_bot - DCb[m][0]]])
    else:
        b[m] = np.array([[- DCb[m][1] / 2, -lenrf_bot],
                         [+ DCb[m][1] / 2, -lenrf_bot],
                         [+ DCb[m][1] / 2, -lenrf_bot - DCb[m][0]],
                         [- DCb[m][1] / 2, -lenrf_bot - DCb[m][0]]])

    # Adding DCs. The same algorythm exept sign y-ax
    for k in range(m, nb - 1):
        b[k + 1] = np.array([b[k][1] + np.array([0, 0]), b[k][1] + np.array([DCb[k + 1][1], 0]),
                             b[k][2] +
                             np.array(
                                 [DCb[k + 1][1], -DCb[k + 1][0] + DCb[k][0]]),
                             b[k][2] + np.array([0, -DCb[k + 1][0] + DCb[k][0]])])

    for k in range(1, m + 1):
        r = m - k
        b[r] = np.array([b[r + 1][0] + np.array([- DCb[r][1], 0]), b[r + 1][0] + np.array([0, 0]),
                         b[r + 1][3] +
                         np.array([0, -DCb[r][0] + DCb[r + 1][0]]),
                         b[r + 1][3] + np.array([- DCb[r][1], -DCb[r][0] + DCb[r + 1][0]])])

    for i in range(n):
        t[i] = t[i][::-1]
    for i in range(nb):
        b[i] = b[i][::-1]

    # Creating array of electrodes with names
    for i in range(n):
        st = "t[" + str(i + 1) + "]"
        electrodes.append([st, [t[i]]])
    for i in range(nb):
        st = "b[" + str(i + 1) + "]"
        electrodes.append([st, [b[i]]])
    electrodes.append(["c", [c]])

    # Polygon approach. All DCs are 0 for now
    s = System([PolygonPixelElectrode(cover_height=cheight, cover_nmax=cmax, name=n, paths=map(np.array, p))
                for n, p in electrodes])
    for i in range(n_rf):
        st = 'rf' + str(i) + 'u'
        s[st].rf = Urf[i][0]
        st = 'rf' + str(i) + 'd'
        s[st].rf = Urf[i][1]

    for i in range(n):
        st = "t[" + str(i + 1) + "]"
        s[st].dc = 0
    for i in range(nb):
        st = "b[" + str(i + 1) + "]"
        s[st].dc = 0
    s["c"].dc = 0
    DC.extend(np.array(t)*L)
    DC.extend(np.array(b)*L)
    DC.append(np.array(c)*L)

    # creates a plot of electrode
    if need_plot:
        fig, ax = plt.subplots(1, 2, figsize=(60, 60))
        s.plot(ax[0])
        s.plot_voltages(ax[1], u=s.rfs)
        # u = s.rfs sets the voltage-type for the voltage plot to RF-voltages (DC are not shown)
        xmax = rflength * 2 / 3
        ymaxp = (np.max(DCtop) + lenrf_top + cwidth) * 1.2
        ymaxn = (np.max(DCbottom) + lenrf_bot) * 1.2
        ax[0].set_title("colour")
        # ax[0] addresses the first plot in the subplots - set_title gives this plot a title
        ax[1].set_title("rf-voltages")
        for axi in ax.flat:
            axi.set_aspect("equal")
            axi.set_xlim(-xmax, xmax)
            axi.set_ylim(-ymaxn, ymaxp)
        if save_plot:
            plt.tight_layout()
            plt.savefig(save_plot)
            
    if need_coordinates:
        return s, RF, DC
    else:
        return s

def polygons_from_gds(gds_lib, L = 1e-6, need_plot = True, need_coordinates = True, cheight=0, cmax=0, save_plot = None):
    '''
    Creates polygon trap from GDS file, consisting this trap. 
    It then plots the trap with indexes for each electrode, 
    for convenient defining of all voltages.

    Parameters
    ----------
    gds_lib : file.GDS
        file with GDS structure. This function will read only the top layer.
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.
    need_plot : bool, optional
        If True, returns a plot of the trap with each electrode assigned its index 
        along the order from the GDS file. The default is False.
    need_coordinates : bool, optional
        If True, returns the coordinates, scaled by L to SU form, used by polygon_simulation(). The default is False.
    cheight : float, optional
        Height of cover electrode - grounded plane above the trap. The default is 0.
    cmax : int, optional
        Expansion order for cover electrode. Usually 5 is sufficient for correct calculations. The default is 0.
    save_plot : str, optional
        Saves the plot to the file for the provided path.
        For example: "five-wire_trap.eps" will create this file with the plot.
        The default is None.

    Returns
    -------
    s : electrode.System object
        Surface trap from electrode .
    full_elec : list shape (number of electrodes, electrode shape), optional
        returns electrodes coordinates in SU for MD simulation or additional reordering.


    '''
    lib = gdspy.GdsLibrary(infile=gds_lib)
    count = 0
    full_elec = []
    electrodes = []

    for cell in lib.top_level():
        for coordinates in cell.get_polygons():
            electrodes.append([f'[{count}]', [coordinates[::-1]]])
            count+=1
            full_elec.append(np.array(coordinates[::-1])*L)
    s = System([PolygonPixelElectrode(cover_height=cheight, cover_nmax=cmax, name=n, paths=map(np.array, p))
                for n, p in electrodes]) 
    
    # creates a plot of electrode
    if need_plot:
        fig, ax = plt.subplots(1, 1, figsize = [30, 30])
        s.plot(ax)
        ax.set_title("electrode layout")
        ymaxes = []
        ymines = []
        xmaxes = []
        xmines = []
        for elec in full_elec:
            ymaxes.append(np.max(elec[:,1]/L))
            ymines.append(np.min(elec[:,1]/L))
            xmaxes.append(np.max(elec[:,0]/L))
            xmines.append(np.min(elec[:,0]/L))
        ax.set_xlim(np.min([1.2*np.min(xmines), 0.8*np.min(xmines)]), np.max([1.2*np.max(xmaxes), 0.8*np.max(xmaxes)]))
        ax.set_ylim(np.min([1.2*np.min(ymines), 0.8*np.min(ymines)]), np.max([1.2*np.max(ymaxes), 0.8*np.max(ymaxes)]))
        if save_plot:
            plt.tight_layout()
            plt.savefig(save_plot)
            
    if need_coordinates:
        return s, full_elec
    else:
        return s
    

def polygons_reshape(full_electrode_list, order, L = 1e-6, need_plot = True, need_coordinates = True, cheight=0, cmax=0, save_plot = None):
    '''
    Sometimes it is convenient to have specific order of electrodes: RF starting first and so on.
    This function will reorder the electrode array, obtained from polygon_to_gds(),
    for a desired order.

    Parameters
    ----------
    full_electrode_list : list shape (number of electrodes, electrode shape)
        electrodes coordinates in SU from polygon_to_gds().
    order : list shape (number of electrodes)
        Desired order, along which electrode indeces will be reassigned.
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.
    need_plot : bool, optional
        If True, returns a plot of the trap with each electrode assigned its index 
        along the order from the GDS file. The default is False.
    need_coordinates : bool, optional
        If True, returns the coordinates, scaled by L to SU form, used by polygon_simulation(). The default is False.
    cheight : float, optional
        Height of cover electrode - grounded plane above the trap. The default is 0.
    cmax : int, optional
        Expansion order for cover electrode. Usually 5 is sufficient for correct calculations. The default is 0.
    save_plot : str, optional
        Saves the plot to the file for the provided path.
        For example: "five-wire_trap.eps" will create this file with the plot.
        The default is None.

    Returns
    -------
    s : electrode.System object
        Surface trap from electrode .
    full_elec : list shape (number of electrodes, electrode shape), optional
        returns electrodes coordinates in SU for MD simulation or additional reordering.

    '''
    electrodes = []
    full_elec = []
    
    for i, elec in enumerate(order):
        electrodes.append([f'[{i}]', [full_electrode_list[elec]/L]])
        full_elec.append(np.array(full_electrode_list[elec]))
    s = System([PolygonPixelElectrode(cover_height=cheight, cover_nmax=cmax, name=n, paths=map(np.array, p))
                for n, p in electrodes]) 
    
    # creates a plot of electrode
    if need_plot:
        fig, ax = plt.subplots(1, 1, figsize = [30, 30])
        s.plot(ax)
        ax.set_title("electrode layout")
        ymaxes = []
        ymines = []
        xmaxes = []
        xmines = []
        for elec in full_elec:
            ymaxes.append(np.max(elec[:,1]/L))
            ymines.append(np.min(elec[:,1]/L))
            xmaxes.append(np.max(elec[:,0]/L))
            xmines.append(np.min(elec[:,0]/L))
        ax.set_xlim(np.min([1.2*np.min(xmines), 0.8*np.min(xmines)]), np.max([1.2*np.max(xmaxes), 0.8*np.max(xmaxes)]))
        ax.set_ylim(np.min([1.2*np.min(ymines), 0.8*np.min(ymines)]), np.max([1.2*np.max(ymaxes), 0.8*np.max(ymaxes)]))
        if save_plot:
            plt.tight_layout()
            plt.savefig(save_plot)
            
    if need_coordinates:
        return s, full_elec
    else:
        return s
    
def gapping(elec, gap):
    '''
    Shrinks polygonal electrode by the "gap" parameter, creating gaps between electrodes

    Parameters
    ----------
    elec : np.array shape [number of points in electrode, 2]
        Electrode to shrink, defined by its coordinates.
    gap : float
        Length to shrink the electrode. Corresponds to the width of the gap /2.

    Returns
    -------
    gapped : np.array shape [number of points in electrode, 2]
        Shrinked electrode.

    '''
    poly = Polygon(elec)
    gapped = []
    newel = np.concatenate([[elec[-1]], elec, [elec[0]]])
    for i, el in enumerate(newel):
        try:
            prev = el-newel[i-1]
            nest = newel[i+1]-el
        except:
            pass
        if nest[0]>0 and prev[1]<0:
            dot1 = el + np.array([gap, gap])
            dot2 = el - np.array([gap, gap])
            point = Point(dot1[0], dot1[1])
            if poly.contains(point):
                gapped.append(dot1)
            else:
                gapped.append(dot2)
        if nest[1]>0 and prev[0]>0:
            dot1 = el + np.array([-gap, gap])
            dot2 = el - np.array([-gap, gap])
            point = Point(dot1[0], dot1[1])
            if poly.contains(point):
                gapped.append(dot1)
            else:
                gapped.append(dot2)
        if nest[0]<0 and prev[1]>0:
            dot1 = el + np.array([-gap, -gap])
            dot2 = el - np.array([-gap, -gap])
            point = Point(dot1[0], dot1[1])
            if poly.contains(point):
                gapped.append(dot1)
            else:
                gapped.append(dot2)
        if nest[1]<0 and prev[0]<0:
            dot1 = el + np.array([gap, -gap])
            dot2 = el - np.array([gap, -gap])
            point = Point(dot1[0], dot1[1])
            if poly.contains(point):
                gapped.append(dot1)
            else:
                gapped.append(dot2)
        if nest[0]>0 and prev[1]>0:
            dot1 = el + np.array([-gap, gap])
            dot2 = el - np.array([-gap, gap])
            point = Point(dot1[0], dot1[1])
            if poly.contains(point):
                gapped.append(dot1)
            else:
                gapped.append(dot2)
        if nest[1]>0 and prev[0]<0:
            dot1 = el + np.array([-gap, -gap])
            dot2 = el - np.array([-gap, -gap])
            point = Point(dot1[0], dot1[1])
            if poly.contains(point):
                gapped.append(dot1)
            else:
                gapped.append(dot2)
        if nest[0]<0 and prev[1]<0:
            dot1 = el + np.array([-gap, gap])
            dot2 = el - np.array([-gap, gap])
            point = Point(dot1[0], dot1[1])
            if poly.contains(point):
                gapped.append(dot1)
            else:
                gapped.append(dot2)
        if nest[1]<0 and prev[0]>0:
            dot1 = el + np.array([gap, gap])
            dot2 = el - np.array([gap, gap])
            point = Point(dot1[0], dot1[1])
            if poly.contains(point):
                gapped.append(dot1)
            else:
                gapped.append(dot2)
                
    gapped = np.array(gapped)
    return gapped

def polygon_to_gds(trap, name, gap = 0):
    '''
    Creates GDS file of the trap with a given (uniformal) gap between electrodes

    Parameters
    ----------
    trap : list shape [ number of electrodes, electrode shape]
        List of electrode coordinates for each electrode in the trap. Must be 
        in um scale.
    name : str
        Name of the file to create GDS. Must end with '.gds'.
    gap : float, optional
        Gap width between two electrodes. The default is 0.

    Returns
    -------
    None.

    '''
    lib = gdspy.GdsLibrary()
    try:
        r = np.random.choice(1000, 1)
        cell = lib.new_cell(f'gold{r}')
    except:
        r = np.random.choice(1000, 1)
        cell = lib.new_cell(f'gold{r}')
    for rect in trap:
        rec = gapping(rect, gap/2)
        poly = gdspy.Polygon(rec)                          
        cell.add(poly)
    lib.write_gds(name)
    pass
    

"""
Useful tools for initializing the simulation
"""

def ioncloud_min(x, number, radius):
    '''
    Assignes ions random positions in some radius from the provided point.

    Parameters
    ----------
    x : np.array shape (3)
        (x, y, z) coordinates of a central point of ion cloud.
    number : int
        Number of ions.
    radius : float
        Radius of a cloud.

    Returns
    -------
    positions : np.array shape (number, 3)
        positions of each ion.

    '''
    positions = []

    for ind in range(number):
        d = np.random.random() * radius
        a = np.pi * np.random.random()
        b = 2 * np.pi * np.random.random()

        positions.append([d * np.sin(a) * np.cos(b) + x[0],
                          d * np.sin(a) * np.sin(b) + x[1],
                          d * np.cos(a) + x[2]])

    return positions


def ions_in_order(x, number, dist):
    '''
    Assigns ions positions in axial linear ion chain with given distance from given point.

    Parameters
    ----------
    x : np.array shape (3)
        (x, y, z) coordinates of a starting point of ion chain.
    number : int
        Number of ions.
    dist : float
        distance between ions in the chain.

    Returns
    -------
    positions : np.array shape (number, 3)
        positions of each ion.

    '''
    positions = []
    xini = x - np.array([dist * number / 2, 0, 0])
    for ind in range(number):
        xi = xini + np.array([dist * ind, 0, 0])
        positions.append(xi)

    return positions

"""
Normal mode calculation
"""

def mathieu_solver(r, *a):
    """
    !!! This function is copied from the electrode package without changes. It is required here to avoid deprecation with numpy > 1.20. !!!

    Solve the generalized Mathieu/Floquet equation::

        x'' + (a_0 + 2 a_1 cos(2 t) + 2 a_2 cos(4 t) ... ) x = 0

    .. math:: \\frac{\\partial^2x}{\\partial t^2} + \\left(a_0 + \\sum_{i=1}^k
        2 a_i \\cos(2 i t)\\right)x = 0

    in n dimensions.

    Parameters
    ----------
    r : int
        frequency cutoff at `+- r`
    *a : tuple of array_like, all shape (n, n)
        `a[0]` is usually called `q`.
        `a[1]` is often called `-a`.
        `a[i]` is the prefactor of the `2 cos(2 i t)` term.
        Each `a[i]` can be an (n, n) matrix. In this case the `x` is an
        (n,) vector.

    Returns
    -------
    mu : array, shape (2*n*(2*r + 1),)
        eigenvalues
    b : array, shape (2*r + 1, 2, n, 2*n*(2*r + 1))
        eigenvectors with the following indices:
        (frequency component (-r...r), derivative, dimension,
        eigenvalue). b[..., i] the eigenvector to the eigenvalue mu[i].

    Notes
    -----
    * the eigenvalues and eigenvectors are not necessarily ordered
      (see numpy.linalg.eig())
    """
    n = a[0].shape[0]
    m = np.zeros((2*r+1, 2*r+1, 2, 2, n, n), dtype=complex)
    for l in range(2*r+1):
        # derivative on the diagonal
        m[l, l, 0, 0] = m[l, l, 1, 1] = np.identity(n)*2j*(l-r)
        # the off-diagonal 1st-1st derivative link
        m[l, l, 0, 1] = np.identity(n)
        # a_0, a_1... on the 2nd-0th component
        # fill to the right and below instead of left and right and left
        for i, ai in enumerate(a):
            if l+i < 2*r+1: # cutoff
                # i=0 (a_0) written twice (no factor of two in diff eq)
                m[l, l+i, 1, 0] = -ai
                m[l+i, l, 1, 0] = -ai
    # fold frequency components, derivative index and dimensions into
    # one axis each
    m = m.transpose((0, 2, 4, 1, 3, 5)).reshape((2*r+1)*2*n, -1)
    mu, b = np.linalg.eig(m)
    # b = b.reshape((2*r+1, 2, n, -1))
    return mu, b

def single_ion_modes(s, potential_minimum, ion_mass, L=1e-6, charge=1, mathieu=False, r=2, Omega=None):
    '''
    Small *electrode* wrapper for more convenient trap frequency calculation. Calculates either secular modes of mathieu modes for a single ion.

    Parameters
    ----------
    s : electrode.System object
        Surface trap from *electrode*.
    potential_minimum : np.array shape (3)
        Potential minimum position for this trap.
    ion_mass : float
        Ion mass in standard units.
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.
    charge : int, optional
        Ion charged, scaled to elementary charge, by default 1.
    mathieu : bool, optional
        If True -- calculates Mathieu modes, if False -- calculates secular modes. By default False
    r : int, optional
        Band cutoff for Mathieu modes calculation, by default 2.
    Omega : float, optional
        If not None -- RF frequency of the trap, by default None. Required for Mathieu modes calculation.

    Returns
    -------
    omegas : np.array shape(3)
        Single ion frequencies in Hz.
    ion_modes : np.array shape([3,3])
        Single ion modes.
    '''
    curv_z, ion_modes = s.modes(potential_minimum, sorted=False)
    omegas = np.sqrt(charge * ct.e * curv_z / ion_mass) / (L * 2 * np.pi)

    if mathieu:
        if Omega is None:
            warnings.warn("Trap radio frequency must be provided for Mathieu modes calculation.")

        scale = np.sqrt(charge * ct.e / ion_mass) / (2 * L * Omega)
        a = 16*scale**2*s.electrical_potential(potential_minimum, "dc", 2, expand=True)[0]
        q = 8*scale*s.electrical_potential(potential_minimum, "rf", 2, expand=True)[0]
        mu, b = mathieu_solver(r, a, q)
        i = mu.imag >= 0
        mu, b = mu[i], b[:, i]
        i = mu.imag.argsort()
        mu, b = mu[i], b[:, i]
        omegas = mu[:3].imag * Omega / (2 * 2 * np.pi)
        ion_modes = b[len(b)//2 - 3:len(b)//2, :3].real.T
        
    return omegas, ion_modes

def equilibrium_ion_positions(s, dc_set, ion_masses, ion_number, potential_minimum=None, L = 1e-6, charges = 1, positions_guess=None):
    '''
    Calculates the equilibrium ion positions in a crystal in pseudopotential approximation by minimizing the system's energy.

    Parameters
    ----------
    s : electrode.System object
        Surface trap from electrode .
    dc_set : array shape (len(DCs))
        Set of the voltages on DC electrodes.
    ion_masses : np.array shape (ion number) or float
        Either array of ion masses for mixed species crystal, or a single ion mass for a single species crystal
    ion_number : int
        Ion number
    potential_minimum : np.array shape (3), optional
        If not None -- guess for the potential minimum position. Not needed, if positions_guess is not None.
        by default None. 
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.
    charges : np.array shape (ion_number) or int, optional
        Array of ion charges for crystals with varying charged, or a single charge for uniformly charged ions. Scaled for elementary charge. The default is +1.
    positions_guess : np.array shape(ion_number, 3), optional
        If not None -- initial guess for ion positions in the crystal. If None, potential_minimum should be not None.
        by default None

    Returns
    -------
    equilibrium positions : np.array shape(ion_number, 3)
        Array of equilibrium ion positions in the surface trap. 
    '''
    N = ion_number
    try:
        charges[0]
    except:
        charges = np.ones(N)*charges
    try:
        ion_masses[0]
    except:
        ion_masses = np.ones(N)*ion_masses
    rf_set = s.rfs[s.rfs != 0]
    u_set = np.concatenate([np.zeros(len(rf_set)), dc_set])
    
    def system_potential(x):
        '''
        Returns potential energy of the ion crystal in surface trap

        Parameters
        ----------
        x : np.array shape(3 * ion_number)
            concatentated array of ion coordinates

        Returns
        -------
        pot : float
            Potential energy in eV
        '''
        x = np.array_split(x, N)
        with s.with_voltages(dcs = u_set, rfs = None):
            pot = 0
            for i, x0 in enumerate(x):
                pot += s.potential(x0)
                for j in range(i+1,N):
                    kap = ct.e*charges[i]*charges[j] / (4*np.pi*ct.epsilon_0)/L
                    pot += kap / ((x0[0] - x[j][0])**2 + (x0[1] - x[j][1])**2 + (x0[2] - x[j][2])**2) ** (0.5)
        return pot
    
    if positions_guess is None:
        with s.with_voltages(dcs = u_set, rfs = None):
            if potential_minimum is None:
                warnings.warn("Results may be wrong: potential_minimum guess is not provided.")
                potential_minimum = s.minimum(np.array([0, 1, 2]), method='Newton-CG')

            x_min = s.minimum(potential_minimum, method='Newton-CG')
            curv_z, mod_dir = s.modes(x_min, sorted=False)
            omega_sec=np.sqrt(ct.e*curv_z/ion_masses[0])/(L)
            l = (ct.e**2/(4*np.pi*ct.epsilon_0*ion_masses[0]*omega_sec[0]**2))**(1/3)/L
            positions_guess = np.zeros([N,3])

            for i in range(N):
                positions_guess[i] = x_min - np.array([l*(N-1)/2, 0, 0]) + np.array([i*l,0,0])

    positions_guess = np.concatenate(positions_guess)
    res = minimize(system_potential, positions_guess, method='L-BFGS-B', tol = 1e-16)
    equilibrium_positions = np.array(np.array_split(res.x, N))

    return equilibrium_positions

def hessian(ion_positions, omega_sec, ion_masses, charges):
    '''
    Hessian of harmonic ion potential for ion crystal, necessary to calculate normal modes.
    Is a hessian for N ion modes and 3 principle axes, defined in the following form:
        [[XX XY XZ]
         [YX YY YZ]
         [ZX ZY ZZ]], XX = matrix (N*N)

    Parameters
    ----------
    ion_positions : np.array shape (ion number, 3)
        Array of equilibrium ion positions.
    omega_sec : np.array shape (ion number, 3)
        Array of secular frequencies in 3 axes for each ion.
    ion_masses : np.array shape (ion number)
        Array of ion masses.
    charges : np.array shape (ion number)
        Array of ion charges.

    Returns
    -------
    A_matrix : np.array(3*ion_number, 3*ion_number)
        Hessian.
    M_matrix : np.array(3*ion_number, 3*ion_number)
        Mass matrix, used to retrieve normal modes.

    '''
    global amu, ech, eps0
    ion_positions = np.array(ion_positions)
    omega_sec = np.array(omega_sec)*2*np.pi
    l = (ech**2 / (4*np.pi*eps0*ion_masses[0]*(omega_sec[0][2])**2))**(1/3)
    ion_positions = ion_positions/l
    N = len([len(a) for a in ion_positions])
    M_matrix = np.diag(list(np.array(ion_masses)**(-0.5))*3)

    def d(i, j):
        return np.linalg.norm(ion_positions[i] - ion_positions[j])/(charges[i]*charges[j])
    a1 = []
    for i in range(N):
        alpha = (ion_masses[i]*omega_sec[i][0]**2) / \
            (ion_masses[0]*omega_sec[0][2]**2)
        for p in range(N):
            if i != p:
                alpha += -1 / \
                    d(i, p)**3+3*(ion_positions[i][0] -
                                  ion_positions[p][0])**2/(d(i, p)**5)
        a1.append(alpha)
    for i in range(N):
        alpha = (ion_masses[i]*omega_sec[i][1]**2) / \
            (ion_masses[0]*omega_sec[0][2]**2)
        for p in range(N):
            if i != p:
                alpha += -1 / \
                    d(i, p)**3+3*(ion_positions[i][1] -
                                  ion_positions[p][1])**2/(d(i, p)**5)
        a1.append(alpha)
    for i in range(N):
        alpha = (ion_masses[i]*omega_sec[i][2]**2) / \
            (ion_masses[0]*omega_sec[0][2]**2)
        for p in range(N):
            if i != p:
                alpha += -1 / \
                    d(i, p)**3+3*(ion_positions[i][2] -
                                  ion_positions[p][2])**2/(d(i, p)**5)
        a1.append(alpha)
    A_matrix = np.diag(a1)

    for i in range(N):
        for j in range(N):
            if j != i:
                A_matrix[i][j] = 1/d(i, j)**3 - 3*(ion_positions[i]
                                                   [0] - ion_positions[j][0])**2/d(i, j)**5
                A_matrix[N+i][N+j] = 1/d(i, j)**3 - 3*(ion_positions[i]
                                                       [1] - ion_positions[j][1])**2/d(i, j)**5
                A_matrix[2*N+i][2*N+j] = 1/d(i, j)**3 - 3*(
                    ion_positions[i][2] - ion_positions[j][2])**2/d(i, j)**5
                A_matrix[i][N+j] = 3*(ion_positions[i][0] - ion_positions[j][0])*(
                    ion_positions[j][1] - ion_positions[i][1])/d(i, j)**5
                A_matrix[i+N][j] = 3*(ion_positions[i][0] - ion_positions[j][0])*(
                    ion_positions[j][1] - ion_positions[i][1])/d(i, j)**5

                A_matrix[i][2*N + j] = 3 * (ion_positions[i][0] - ion_positions[j][0]) * (
                    ion_positions[j][2] - ion_positions[i][2]) / d(i, j) ** 5
                A_matrix[2 * N + i][j] = 3 * (ion_positions[i][0] - ion_positions[j][0]) * (
                    ion_positions[j][2] - ion_positions[i][2]) / d(i, j) ** 5

                A_matrix[2*N+i][N + j] = 3 * (ion_positions[i][2] - ion_positions[j][2]) * (
                    ion_positions[j][1] - ion_positions[i][1]) / d(i, j) ** 5
                A_matrix[N + i][2*N + j] = 3 * (ion_positions[i][2] - ion_positions[j][2]) * (
                    ion_positions[j][1] - ion_positions[i][1]) / d(i, j) ** 5

        A1 = 0
        A2 = 0
        A3 = 0

        for p in range(N):
            if i != p:
                A1 += 3*(ion_positions[i][0] - ion_positions[p][0]) * \
                    (ion_positions[i][1] - ion_positions[p][1])/d(i, p)**5
                A2 += 3 * (ion_positions[i][0] - ion_positions[p][0]) * (
                    ion_positions[i][2] - ion_positions[p][2]) / d(i, p) ** 5
                A3 += 3 * (ion_positions[i][2] - ion_positions[p][2]) * (
                    ion_positions[i][1] - ion_positions[p][1]) / d(i, p) ** 5

        A_matrix[i][N+i] = A1
        A_matrix[N+i][i] = A1
        A_matrix[i][2*N+i] = A2
        A_matrix[2*N+i][i] = A2
        A_matrix[N+i][2*N+i] = A3
        A_matrix[2*N+i][N+i] = A3

    A_matrix = A_matrix*ion_masses[0]*omega_sec[0][2]**2

    return A_matrix, M_matrix


def normal_modes(ion_positions, omega_sec, ion_masses, charges = 1, linear = False, reshape = True):
    '''
    Calculation of normal modes for an arbitrary configuration of ion crystal
    in harmonic potential. Arbitrary means spatially linear or nonlinear (2D, 3D) 
    crystals, of single or mixed-species ions, in individual potential 
    configurations (optical tweezers, individual micro traps), with arbitrary charges.
    This function works not only for surface traps, but for any trap, as long as 
    necessary parameters are provided.

    Parameters
    ----------
    ion_positions : np.array shape (ion number, 3)
        Array of equilibrium ion positions.
    omega_sec : np.array shape (ion number, 3) or (3)
        Array of secular frequencies in 3 axes for each ion. Or a single 
        secular frequency, if they are equal for each ion
    ion_masses : np.array shape (ion number) or float
        Array of ion masses. Or a single mass, if all the ions have equal mass.
    charges : np.array shape (ion_number) or int, optional
        Array of ion charges for crystals with varying charged, or a single charge for uniformly charged ions. Scaled for elementary charge. The default is +1.
    linear : bool, optional
        If True it will return only normal modes for each of principle axes of
        oscillation (x,y,z). correct for linear ion chains. The default is False.
    reshape : bool, optional
        If True, result reshapes so that first come the modes, where maximal value
        of mode vector is in x direction (ie X modes), then Y modes, then Z modes. The default is True.

    Returns
    -------
    norm_freqs : np.array shape (3*ion number)
        Array of normal mode frequencies for each mode, in Hz (so scaled by 2pi).
    norm_modes : np.array shape (3*ion number, 3*ion_number)
        Normal mode matrix for every principle axis of oscillation. Each mode 
        has the following structure: the value of mode vector is given along x(y,z)-axis
        [x_axis[0],...x_axis[ion number], y_axis[0],...y_axis[ion number], z_axis[0],...z_axis[ion number]]
        Here the axes correspond to the directions of secular frequencies (of the first ion in crystal).
        
    Optional returns in case of linear chain
    ----------------------------------------
    norm_freqs : np.array shape (3, ion number)
        Array of normal mode frequencies for each principle axis (x,y,z),
        in Hz (so scaled by 2pi).
    norm_modes : np.array shape (3, ion number, ion_number)
        Normal mode matrix for each axis (x,y,z).

    '''
    N = len([len(a) for a in ion_positions])
    try:
        charges[0]
    except:
        charges = np.ones(N)*charges
    try:
        ion_masses[0]
    except:
        ion_masses = np.ones(N)*ion_masses
    try:
        omega_sec[0][0]
    except:
        omega_sec = np.array([omega_sec for i in range(N)])
    A_matrix, M_matrix = hessian(ion_positions, omega_sec, ion_masses, charges)
    AA = np.dot(M_matrix, np.dot(A_matrix, M_matrix))

    eig, norm_modes = np.linalg.eigh(AA)
    norm_modes = -norm_modes.T
    norm_freqs = np.sqrt(eig)/(2*np.pi)
    if reshape:
        norm_freqs, norm_modes = reshape_modes(N, norm_modes, norm_freqs)
    if linear:
        norm_freqs, norm_modes = reshape_modes(N, norm_modes, norm_freqs)
        x_freqs = norm_freqs[:N]
        y_freqs = norm_freqs[N:2*N]
        z_freqs = norm_freqs[2*N:]
        x_modes = np.zeros([N,N])
        y_modes = np.zeros([N,N])
        z_modes = np.zeros([N,N])
        for i in range(N):
            for j in range(N):
                x_modes[i, j] = norm_modes[i+0*N][j+0*N]
        for i in range(N):
            for j in range(N):
                y_modes[i, j] = norm_modes[i+1*N][j+1*N]
        for i in range(N):
            for j in range(N):
                z_modes[i, j] = norm_modes[i+2*N][j+2*N]
        return np.array([x_freqs, y_freqs, z_freqs]), np.array([x_modes, y_modes, z_modes])
    else:    
        return norm_freqs, norm_modes

def reshape_modes(ion_number, harm_modes, harm_freqs):
    '''
    Reshapes 3N normal mode matrix, so that they come in the order of principle axis.
    Needed, because initially eigenvalues are sorted in ascending order.

    Parameters
    ----------
    ion_number : int
        Number of ions.
    harm_modes : np.array shape (3*ion number, 3*ion_number)
        Normal mode matrix for every principle axis of oscillation.
    harm_freqs : np.array shape (3*ion number)
        Array of normal mode frequencies for each mode.

    Returns
    -------
    harm_modes : np.array shape (3*ion number, 3*ion_number)
        Reordered normal mode matrix for every principle axis of oscillation.
    harm_freqs : np.array shape (3*ion number)
        Reorder array of normal mode frequencies for each mode.

    '''
    x_indx = []
    y_indx = []
    z_indx = []

    for i in range(3*ion_number):
        k = np.argmax(np.abs(harm_modes[i]))
        if  k>=2*ion_number:
            z_indx.append(i)
        if  k<2*ion_number and k>=1*ion_number:
            y_indx.append(i)
        if  k<1*ion_number:
            x_indx.append(i)
    indx = np.concatenate([x_indx, y_indx, z_indx])
    harm_freqs = harm_freqs[indx]
    harm_modes = harm_modes[indx][:]
    return harm_freqs, harm_modes

"""
Anharmonic Mathieu modes
"""

def coulumb_hessian(ion_positions, charges):
    '''
    Returns Hessian of Coulomb potential for ions near equilibrium postions
    in harmonic approximation.

    Parameters
    ----------
    ion_positions : np.array shape (ion number, 3)
        Array of equilibrium ion positions.
    charges : np.array of ints shape (ion_number)
        Array of ion charges, scaled for elementary charge.

    Returns
    -------
    A_matrix : np.array shape (3*ion number, 3*ion number)
        Coulomb Hessian in SU.

    '''
    global amu, ech, eps0
    ion_positions = np.array(ion_positions)
    kap = ech**2 / (4*np.pi*eps0)
    N = len([len(a) for a in ion_positions])

    def d(i, j):
        return np.linalg.norm(ion_positions[i] - ion_positions[j])/(charges[i]*charges[j])
    a1 = []
    for i in range(N):
        alpha = 0
        for p in range(N):
            if i != p:
                alpha += -1 / \
                    d(i, p)**3+3*(ion_positions[i][0] -
                                  ion_positions[p][0])**2/(d(i, p)**5)
        a1.append(alpha)
    for i in range(N):
        alpha = 0
        for p in range(N):
            if i != p:
                alpha += -1 / \
                    d(i, p)**3+3*(ion_positions[i][1] -
                                  ion_positions[p][1])**2/(d(i, p)**5)
        a1.append(alpha)
    for i in range(N):
        alpha = 0
        for p in range(N):
            if i != p:
                alpha += -1 / \
                    d(i, p)**3+3*(ion_positions[i][2] -
                                  ion_positions[p][2])**2/(d(i, p)**5)
        a1.append(alpha)
    A_matrix = np.diag(a1)

    for i in range(N):
        for j in range(N):
            if j != i:
                A_matrix[i][j] = 1/d(i, j)**3 - 3*(ion_positions[i]
                                                   [0] - ion_positions[j][0])**2/d(i, j)**5
                A_matrix[N+i][N+j] = 1/d(i, j)**3 - 3*(ion_positions[i]
                                                       [1] - ion_positions[j][1])**2/d(i, j)**5
                A_matrix[2*N+i][2*N+j] = 1/d(i, j)**3 - 3*(
                    ion_positions[i][2] - ion_positions[j][2])**2/d(i, j)**5
                A_matrix[i][N+j] = 3*(ion_positions[i][0] - ion_positions[j][0])*(
                    ion_positions[j][1] - ion_positions[i][1])/d(i, j)**5
                A_matrix[i+N][j] = 3*(ion_positions[i][0] - ion_positions[j][0])*(
                    ion_positions[j][1] - ion_positions[i][1])/d(i, j)**5

                A_matrix[i][2*N + j] = 3 * (ion_positions[i][0] - ion_positions[j][0]) * (
                    ion_positions[j][2] - ion_positions[i][2]) / d(i, j) ** 5
                A_matrix[2 * N + i][j] = 3 * (ion_positions[i][0] - ion_positions[j][0]) * (
                    ion_positions[j][2] - ion_positions[i][2]) / d(i, j) ** 5

                A_matrix[2*N+i][N + j] = 3 * (ion_positions[i][2] - ion_positions[j][2]) * (
                    ion_positions[j][1] - ion_positions[i][1]) / d(i, j) ** 5
                A_matrix[N + i][2*N + j] = 3 * (ion_positions[i][2] - ion_positions[j][2]) * (
                    ion_positions[j][1] - ion_positions[i][1]) / d(i, j) ** 5

        A1 = 0
        A2 = 0
        A3 = 0

        for p in range(N):
            if i != p:
                A1 += 3*(ion_positions[i][0] - ion_positions[p][0]) * \
                    (ion_positions[i][1] - ion_positions[p][1])/d(i, p)**5
                A2 += 3 * (ion_positions[i][0] - ion_positions[p][0]) * (
                    ion_positions[i][2] - ion_positions[p][2]) / d(i, p) ** 5
                A3 += 3 * (ion_positions[i][2] - ion_positions[p][2]) * (
                    ion_positions[i][1] - ion_positions[p][1]) / d(i, p) ** 5

        A_matrix[i][N+i] = A1
        A_matrix[N+i][i] = A1
        A_matrix[i][2*N+i] = A2
        A_matrix[2*N+i][i] = A2
        A_matrix[N+i][2*N+i] = A3
        A_matrix[2*N+i][N+i] = A3

    return A_matrix.T*kap

def trap_hessian(ion_positions, s, rf_set, Omega, dc_set, ion_masses, charges, L, N, alpha, alpha2):
    '''
    Returns anharmonic Hessian of the surface trap potential. Returns hessians for DC and RF field, respectively. 
    The following form, compatible with Coulomb Hessian, is used:
        [[XX XY XZ]
         [YX YY YZ]
         [ZX ZY ZZ]], where each
        
        XX = [[U_xx, 0, .., 0] 
              [0, U_xx, .., 0] 
              ...
              [0, .., 0, U_xx]],
        XY = [[U_xy, 0, .., 0] 
              [0, U_xy, .., 0] 
              ...
              [0, .., 0, U_xy]], ...
        is an ion number * ion number diagonal matrix.

    Parameters
    ----------
    ion_positions : np.array shape (ion number, 3)
        Array of equilibrium ion positions.
    s : electrode.System object
        Surface trap from electrode package
    rf_set : list shape (RFs)
        List of RF amplitudes for each electrode in SU.
    Omega : float
        RF frequency of the trap
    dc_set : array shape (len(DCs))
        Set of the voltages on DC electrodes.
    ion_masses : np.array shape (ion number)
        Array of ion masses.
    charges : np.array of ints shape (ion_number)
        Array of ion charges, scaled for elementary charge.
    L : float
        Length scale of the electrode. 
    N : int
        ion number.
    alpha : float
        Hexapole anharmonic terms coefficient. 0 if anharmonic effects turned off, 1 if turned on
    alpha2 : float
        Octopole anharmonic terms coefficient. 0 if anharmonic effects turned off, 1 if turned on

    Returns
    -------
    A : np.array shape (3*ion number, 3*ion number)
        Anharmonic Hessian of the trap DC potential in SU.
    Q : np.array shape (3*ion number, 3*ion number)
        Anharmonic Hessian of the trap RF potential in SU.
    M_matrix : np.array(3*ion_number, 3*ion_number)
        Mass matrix, used to retrieve normal modes for single and mixed species crystals.

    '''
    u_set = np.concatenate((np.zeros(len(rf_set)), dc_set))
    rf_set = np.array(rf_set)
    A_hes = []
    Q_hes = []

    for i,pos in enumerate(ion_positions):
        Urf = rf_set*np.sqrt(ech*charges[i]/ion_masses[i])/(2*L*Omega)

        with s.with_voltages(dcs = u_set, rfs = Urf):
            min_pos = np.array(s.minimum( 1.001*pos/L, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG"))

        with s.with_voltages(dcs = u_set, rfs = rf_set):
            A_deriv2 = s.electrical_potential(min_pos, "dc", 2, expand=True)[0]*ech/L**2 
            Q_deriv2 = s.electrical_potential(min_pos, "rf", 2, expand=True)[0]*ech/L**2

            A_deriv3 = np.zeros([3,3])
            Q_deriv3 = np.zeros([3,3])
            pota = s.electrical_potential(min_pos, "dc", 3, expand=True)[0]*ech/L**2 
            potq = s.electrical_potential(min_pos, "rf", 3, expand=True)[0]*ech/L**2
            
            for r in range(N):
                for p in range(3):
                    A_deriv3 = A_deriv3 + 1/3*(ion_positions[r][p]/L-min_pos[p])*pota[p]
                    Q_deriv3 = Q_deriv3 + 1/3*(ion_positions[r][p]/L-min_pos[p])*potq[p]

            A_deriv4 = np.zeros([3,3])
            Q_deriv4 = np.zeros([3,3])
            potaa = s.electrical_potential(min_pos, "dc", 4, expand=True)[0]*ech/L**2
            potqq = s.electrical_potential(min_pos, "rf", 4, expand=True)[0]*ech/L**2

            for r in range(N):
                for v in range(N):
                    for p in range(3):
                        for c in range(3):
                            A_deriv4 = A_deriv4 + 1/12*(ion_positions[r][p]/L-min_pos[p])*(ion_positions[v][c]/L-min_pos[c])*potaa[p][c]
                            Q_deriv4 = Q_deriv4 + 1/12*(ion_positions[r][p]/L-min_pos[p])*(ion_positions[v][c]/L-min_pos[c])*potqq[p][c]

            A_hes.append(A_deriv2 + alpha * A_deriv3 + alpha2 * A_deriv4)
            Q_hes.append(Q_deriv2 + alpha * Q_deriv3 + alpha2 * Q_deriv4)

    A_hes = np.array(A_hes)
    Q_hes = np.array(Q_hes)
    real_hessian_A = np.zeros([3,3,N,N])
    real_hessian_Q = np.zeros([3,3,N,N])

    for p in range(3):
        for c in range(3):
            diag_A = np.array([hess[p,c] for hess in A_hes])
            real_hessian_A[p,c] = np.diag(diag_A)
            diag_Q = np.array([hess[p,c] for hess in Q_hes])
            real_hessian_Q[p,c] = np.diag(diag_Q)
            
    A = np.block([[real_hessian_A[0,0], real_hessian_A[0,1], real_hessian_A[0,2]],
                  [real_hessian_A[1,0], real_hessian_A[1,1], real_hessian_A[1,2]],
                  [real_hessian_A[2,0], real_hessian_A[2,1], real_hessian_A[2,2]]
                 ])
    Q = np.block([[real_hessian_Q[0,0], real_hessian_Q[0,1], real_hessian_Q[0,2]],
                  [real_hessian_Q[1,0], real_hessian_Q[1,1], real_hessian_Q[1,2]],
                  [real_hessian_Q[2,0], real_hessian_Q[2,1], real_hessian_Q[2,2]]
                 ])
    M_matrix = np.diag(list(np.array(ion_masses)**(-0.5))*3)

    return A, Q, M_matrix

def crystal_modes(ion_positions, ion_masses, s, rf_set, Omega, dc_set, 
                  L = 1e-6, charges = 1, reshape=True, anharmonic=True):
    '''
    Returns harmonic modes of an arbitrary ion crystal configuration in surface trap, considering time-dependent 
    potential as in 3N dimensional Mathieu equation and hexapole and octopole anharmonic potential terms of the trap.

    Parameters
    ----------
    ion_positions : np.array shape (ion number, 3)
        Array of equilibrium ion positions.
    ion_masses : np.array shape (ion number) or float
        Array of ion masses. Or a single mass, if all the ions have equal mass.
    s : electrode.System object
        Surface trap from *electrode* package
    rf_set : list shape (RFs)
        List of RF amplitudes for each electrode in SU.
    Omega : float
        RF frequency of the trap
    dc_set : array shape (len(DCs))
        Set of the voltages on DC electrodes.
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.
    charges : np.array of ints shape (ion_number) or int, optional
        Array of ion charges (or single charge), scaled for elementary charge. The default is +1.
    reshape : bool, optional
        If True, result reshapes so that first come the modes, where maximal value
        of mode vector is in x direction (ie X modes), then Y modes, then Z modes. The default is True.
        !!Note may give unexpected results for anharmonic modes, which reflects anharmonic mixing of modes on different
        principle oscillation axes/ !!

    Returns
    -------
    norm_freqs : np.array shape (3*ion number)
        Array of harmonic mode frequencies for each mode, in Hz.
    norm_modes : np.array shape (3*ion number, 3*ion_number)
        Harmonic mode matrix for every principle axis of oscillation with anharmonic 
        modifications for harmonic mode vectors. Each mode has the following structure:
        the value of mode vector is given along x(y,z)-axis
        [x_axis[0],...x_axis[ion number], y_axis[0],...y_axis[ion number], z_axis[0],...z_axis[ion number]]
        Here the axes correspond to the directions of secular frequencies (of the first ion in crystal).

    '''
    N = len([len(a) for a in ion_positions])
    try:
        charges[0]
    except:
        charges = np.ones(N)*charges
    try:
        ion_masses[0]
    except:
        ion_masses = np.ones(N)*ion_masses

    С = coulumb_hessian(ion_positions, charges)

    if anharmonic:
        alpha = 1
        alpha2 = 1
    else:
        alpha = 0
        alpha2 = 0
        
    A, Q, M_matrix = trap_hessian(ion_positions, s, rf_set, Omega, dc_set, ion_masses, charges, L, N, alpha, alpha2)

    scale = 1/((Omega)**2 )
    A = A + С
    A = 4*scale*np.dot(M_matrix, np.dot(A, M_matrix))
    Q = 2*scale*np.dot(M_matrix, np.dot(Q, M_matrix))

    mu, b = mathieu_solver(2, A, Q)
    i = mu.imag >= 0
    mu, b = mu[i], b[:, i]
    i = mu.imag.argsort()
    mu, b = mu[i], b[:, i]
    mu = mu/2
    norm_modes = b[len(b)//2 - 3*N:len(b)//2, :3*N].real
    norm_modes = norm_modes.T
    norm_freqs = mu[:3*N].imag*Omega/(2*np.pi)

    if reshape:
        norm_freqs, norm_modes = reshape_modes(N, norm_modes, norm_freqs)

    return norm_freqs, norm_modes

def anharmonics(s, minimums, axis, L = 1e-6):
    """
    Calculates anharmonic scale lengths at given coordinates.
    The lengthes are given for harmonic, n = 3 and n = 4 terms.
    Length are defined as in DOI 10.1088/1367-2630/13/7/073026 .

    Parameters
    ----------
    s : electrode.System object
        System object from electrode package, defining the trap
    minimums : array shape [number of dots, 3]
        Coordinates of approximate minimum positions (the exact mimums will be
                                                      calculated from them)
    axis : int
        axis, along which the anharmonicity is investigated. 
        Determined by respecting int: (0, 1, 2) = (x, y, z)
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.

    Returns
    -------
    results : list shape [number of dots, 3]
        array of anharmonic scale lengths (in m) in each calculated potential minimum. 
        Given as  [l2, l3, l4], for ln being scale length of the n's potential term

    """
    try:
        minimums[0][0]
    except:
        minimums = [minimums]
    results = []
    for pos in minimums:
        x1 = s.minimum(np.array(pos), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
        deriv2, mod_dir=s.modes(x1,sorted=False)
        deriv3 = np.zeros([3,3])
        for p in range(3):
            pot = s.potential(x1, derivative = 3)[0][p]/L**3
            deriv3 = deriv3 + 1/3*pot
        deriv4 = np.zeros([3,3])
        for p in range(3):
            for c in range(3):
                pot = s.potential(x1, derivative = 4)[0][p][c]/L**4
                deriv4 = deriv4 + 1/12*pot
        
        k2 = deriv2[axis]*ech/L**2
        k3 = np.linalg.eig(deriv3)[0][axis]*ech
        k4 = np.linalg.eig(deriv4)[0][axis]*ech
        l2 = (ech**2/(4*np.pi*eps0*k2))**(1/3)
        l3 = (k3/k2)**(1/(2-3))
        l4 = np.sign(k4)*(np.abs(k4)/k2)**(1/(2-4))
        results.append([l2, l3, l4])
    
    return results


"""
Stability analysis
"""

def stability(s, Ms, Omega, Zs, minimum, L = 1e-6, need_plot = True, save_plot = None):
    """
    Returns stability parameters for the linear planar trap (RF confinement only radial)
    If asked, return plot of the stability a-q diagram for this trap and plots 
    the parameters for this voltage configuration (as a color dot) for given isotopes.
    Additoinally returns geometric parameters of stability for this trap (alpha, theta),
    range of achievable a and q parameters.

    Parameters
    ----------
    s : electrode.System object
        System object from electrode package, defining the trap
    Ms : float or list shape (number of species)
        ion mass in SU, single or an array to plot several isotopes
    Omega : float
        RF frequency 
    Zs : float or list shape (number of species) 
        ion charge in SU, single or an array for different ions
    minimum : list shape (3)
        approximate position of potential minimum, from which the
        real minimum is calculated
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.
    need_plot : int, optional
        if True, then the plot of the stability diagram and the 
        a-q parameters for this voltage configuration is shown. The default is True.
    save_plot : str, optional
        Saves the plot to the file for the provided path.
        For example: "five-wire_trap.eps" will create this file with the plot.
        The default is None.

    Returns
    -------
    params : dictionary
        Stability parameters for this trap, sorted as follows:
            {'Ion (M = mass of this ion, Z = charge of this ion)': {'a': a for this ion, 'q': for this ion},
             .. repeated for all ions,
             'α': alpha parameter,
             'θ': theta parameter,
             'Range of achievable a': list [lower a, upper a],
             'Critical q': maximal q for stability (of the first main stability region)}
    plt.plot : optional
        plot of the stability diagram and trap's a-q parameters on it 

    """
    if type(Ms) is not list: 
        try:
            Ms = list(Ms)
            Zs = list(Zs)
        except:
            Ms = [Ms]
            Zs = [Zs]
            
    params = {}
    for M, Z in zip(Ms, Zs):
        rf_set = s.rfs
        rf_set = rf_set*np.sqrt(Z/M)/(2*L*Omega)
        scale = Z/((L*Omega)**2*M)
        with s.with_voltages(dcs = None, rfs = rf_set):
            x1 = s.minimum(np.array(minimum), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
            #mathieu matrices
            a = 4*scale*s.electrical_potential(x1, "dc", 2, expand=True)[0]  #need to find alpha/alpha' so 2 derivative needed
            q = 4*np.sqrt(scale)*s.electrical_potential(x1, "rf", 2, expand=True)[0]      #so here we obtain hessian of this potential, didj(F)

            a, q = np.array(a), np.array(q)

            #removing axial directions for following stability analisis
            A = a
            Q = q
            A = A[1:3, 1:3]
            Q = Q[1:3, 1:3]

            #diagonalize A
            DiagA, BasA = np.linalg.eig(A)
            DiagA = np.diag(DiagA)
            areal = DiagA[0,0]
            alpha = -DiagA[1,1]/areal

            #diagonalize Q
            DiagQ, BasQ = np.linalg.eig(Q)
            DiagQ = np.diag(DiagQ)
            qreal = DiagQ[0,0]
            params[f'Ion (M = {round(M/amu):d}, Z = {round(Z/ech):d})'] = {'a':areal, 'q':qreal}

    #obtain theta
    rot = np.dot(np.linalg.inv(BasQ),BasA)
    thetha = np.arccos(rot[0,0])
    c = np.cos(2*thetha)
    sin = np.sin(2*thetha)
    params['\u03B1'] = alpha
    params['\u03B8'] = thetha


    #boundaries
    q = np.linspace(0,1.5, 1000)
    ax = -q**2/2
    ab = q**2/(2*alpha)
    ac = 1 - c*q - (c**2/8 + (2*sin**2*(5+alpha))/((1+alpha)*(9+alpha)))*q**2
    ad = -(1 - c*q - (c**2/8 + (2*sin**2*(5+1/alpha))/((1+1/alpha)*(9+1/alpha)))*q**2)/alpha

    #critical a, q
    aa = 1/(2*alpha)
    bb = 1
    cc = -c
    dd = -(c**2/8 + (2*sin**2*(5+alpha))/((1+alpha)*(9+alpha)))
    ee = -1/alpha
    ff = c/alpha
    gg = (c**2/8 + (2*sin**2*(5+1/alpha))/((1+1/alpha)*(9+1/alpha)))/alpha
    hh = -1/2

    q_crit = np.max([(-(cc-ff) - np.sqrt((cc-ff)**2 - 4*(dd-gg)*(bb-ee)))/(2*(dd-gg)), (-(cc-ff) + np.sqrt((cc-ff)**2 - 4*(dd-gg)*(bb-ee)))/(2*(dd-gg))])

    qa_crit_upper = [(-cc - np.sqrt(np.abs(cc**2 - 4*bb*(dd-aa))))/(2*(dd-aa)), (-cc + np.sqrt(np.abs(cc**2 - 4*bb*(dd-aa))))/(2*(dd-aa))]
    qa_crit_lower = [(-ff - np.sqrt(np.abs(ff**2 - 4*ee*(gg-hh))))/(2*(gg-hh)), (-ff + np.sqrt(np.abs(ff**2 - 4*ee*(gg-hh))))/(2*(gg-hh))]
    if qa_crit_upper[0] < q_crit and qa_crit_upper[0] > 0:
        a_crit_upper = qa_crit_upper[0]**2/(2*alpha)
    if qa_crit_upper[1] < q_crit and qa_crit_upper[1] > 0:
        a_crit_upper = qa_crit_upper[1]**2/(2*alpha)
    if qa_crit_lower[0] < q_crit and qa_crit_lower[0] > 0:
        a_crit_lower = -qa_crit_lower[0]**2/2
    if qa_crit_lower[1] < q_crit and qa_crit_lower[1] > 0:
        a_crit_lower = -qa_crit_lower[1]**2/2

    try:
        params['Range of achievable a'] = [a_crit_lower, a_crit_upper]
    except:
        a_crit_lower = -1
        a_crit_upper = 1
    params['Critical q'] = q_crit

    
    if need_plot:
        #plotting boundaries 
        fig = plt.figure()
        fig.set_size_inches(7,5)
        plt.plot(q,ax, 'g')
        plt.plot(q,ab, 'g')
        plt.plot(q,ac, 'g')
        plt.plot(q,ad, 'g')
        
        #unifying noudaries to two united line so the area between them can be filled by plt.fill_between
        y1 = np.array(list(map(max, zip(ax, ad))))
        y2 = np.array(list(map(min, zip(ab, ac))))
        
        y_lim = np.max(np.array([a_crit_upper, -a_crit_lower]))
        
        plt.ylim(-y_lim*1.25, y_lim*1.25)
        plt.xlim(0, q_crit*1.25)
        plt.plot(q,y1, 'g')
        plt.plot(q,y2, 'g')
        plt.xlabel('q', fontsize = '30')
        plt.ylabel('a', fontsize = '30')
        plt.fill_between(q, y1, y2,where=y2>=y1, interpolate = True, color = 'skyblue')
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        plt.locator_params(axis='x', nbins=6)
        plt.locator_params(axis='y', nbins=6)
       
    
        #plottin a and q, obtained from potential hessian 
        colors=["maroon", 'peru',"darkgoldenrod",'magenta', "orangered", 'darkorange', 'crimson', 'brown']
        k = 0
        for M, Z in zip(Ms, Zs):
            plt.scatter(params[f'Ion (M = {round(M/amu):d}, Z = {round(Z/ech):d})']['q'], params[f'Ion (M = {round(M/amu):d}, Z = {round(Z/ech):d})']['a'], s = 40, edgecolor='black', color = colors[k], label = f'Ion (M = {round(M/amu):d}, Z = {round(Z/ech):d})' )
            k = (k+1)%8
        plt.legend()
        plt.tight_layout()
        if save_plot:
            plt.savefig(save_plot)
    
        plt.show()
        
    return params

"""
Optimization functions

The optimization routine is written in Python, which affects its speed.
However, the main source of the slowness of these algorithms is a necessity 
to determine the potential minimum to calculate the loss function. 
The potential minimum is determined with its own optimization, which is already
written in Cython, so only minor increase in speed can be obtained by rewritting these functions in C. 
"""

def v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L, micro):
    """
    Loss function for the DC voltage optimization.
    If the individual potential well is lost, returns 1000
    
    Returns
    -------
    loss : float
        Loss value, calculated as a quadratic norm of difference between the 
        actual and desired set of secular frequencies

    """

    loss = 0
    u_set = np.zeros(len([rfs for rfs in s.rfs if rfs != 0]))
    u_set = np.concatenate([u_set, uset])
    
    #obtain result of the function
    with s.with_voltages(dcs = u_set, rfs = None):
        for i, pos in enumerate(dots):
            try:
                x1 = s.minimum(pos, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
                curv, mod_dir=s.modes(x1,sorted=False) 
                for j, axs in enumerate(axis):
                    omega = (np.sqrt(Z*curv[axs]/M)/(L*2*np.pi) * 1e-6)/omegas[j][0]
                    loss += (omega - omegas[j][i]/omegas[j][0])**2 
                loss += micro*np.linalg.norm(x1-pos)**2

            except: 
                sys.exit("The ion in %s positions is lost." %i)
            
    return loss

def v_stoch_grad(uset, stoch, numbers, s, dots, axis, omegas, Z, M, L, micro):
    """
    Calculates stochastic gradient, where partial derivative is calculated 
    only for "stoch:int" randomly chosen parameters

    Returns
    -------
    np.ndarray(numbers[1])
        Calculated gradient

    """
    rflen = len([rfs for rfs in s.rfs if rfs != 0])
    dclen = len(s.dcs)
    df = np.zeros(dclen-rflen)
    param = np.random.choice(numbers, stoch, replace = 'False')
    param = np.sort(param)
    for i in param:
        uset[i] += 1e-8
        fplus = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L, micro)
        uset[i] -= 2e-8
        fmin = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L, micro)
        uset[i] += 1e-8
        df[i]  = (fplus - fmin)/(2e-8)
        
    return np.array(df)

def v_precise_grad(uset, numbers, s, dots, axis, omegas, Z, M, L, micro):
    """
    Calculates exact gradient of loss function

    Returns
    -------
    np.ndarray(numbers[1])
        Calculated gradient
    """
    
    rflen = len([rfs for rfs in s.rfs if rfs != 0])
    dclen = len(s.dcs)
    df = np.zeros(dclen-rflen)
    
    for i in numbers:
        uset[i] += 1e-8
        fplus = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L, micro)
        uset[i] -= 2e-8
        fmin = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L, micro)
        uset[i] += 1e-8
        df[i]  = (fplus - fmin)/(2e-8)
        
    return np.array(df)



def voltage_optimization(s, Z, M, dots, axis, omegas, start_dcset, learning_rate, numbers = 0, stoch = 0, eps = 1e-8, L = 1e-6, step = 100, micro = False):
    """
    A function, determining the DC voltage set for the trap, at which the set
    of desired secular frequencies is achieved at the requested positions, 
    using the ADAM optimization.
    The final positions of potential minima, however, will slightly change due
    to the change in the DC potential.
    The secular frequencies may be determined in all the required principle 
    axes of oscillation simultaneously.
    The optimization is very slow, and may take hours, so at each number of 
    iterations, determined by the user, it will print the current result.

    Parameters
    ----------
    s : electrode.System object
        Your trap. It is important, that its RF potential is predetermined, 
        and all DC electrodes are grounded, before applying this function.
    Z : float
        Ion charge
    M : float
        Ion mass
    dots : list shape (number of positions, 3)
        Positions, at which secular frequencies are calculated
    axis : list shape (number of axes)
        List of all principle axes, at which the desired secular frequencies
        are determined. Defined as (0, 1, 2) = (x, y, z). For example,
        if axis == (1), only frequencies at y axis are considered.
        If axis == (0, 1, 2), all axes are considered simultaneously
    omegas : list shape (len(axis), number of positions)
        Set of desired secular frequencies in MHz (/2pi), given for 
        all positions and axes. 
    start_dcset : list shape (numbers[1])
        Starting voltages for the optimization. The closer they are, the faster the algorithm
    learning_rate : float
        Learning rate of the algorithm. The optimal learning rate may take 
        various values from 0.000001 to 1000. This should be tested in a particular case.
    numbers : list shape (number of optimizing electrodes) or int, optional
        list of the indeces of DC electrodes, used for optimization. 
        If 0 (default), all DC elecrtodes are used for optimization.
    stoch : int, optional
        The default is 0. This value indicates, if the exact gradient descent 
        (stoch == 0) is performed, or the stochastic gradient descent, with 
        the stochastic choice of "stoch" parameters each iteration.
        The stochastic method is much faster, but for closer voltage sets it
        may loose convergence. In this case, user should switch to the exact gradients.
    eps : float, optional
        Convergence, at each the optimization is stopped. The default is 1e-8.
    L : float, optional
        Dimension scale of the electrode. The default is 1e-6 which means um.
    step : int, optional
        Number of iterations, at which the current voltage set will be printed.
        The default is 100.
    micro : bool, optional
        If True, optimization tries to include micromotion compensation to voltage set.
        The default is False.

    Returns
    -------
    uset : list shape (numbers[1])
        Resulting voltage set.

    """
    uset = np.array(start_dcset)
    if micro:
        mic = 1
    else:
        mic = 0
    try:
        numbers[0]
    except:
        numbers = np.arange(len(uset))
    loss2 =  v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L, mic)
    print("Initial loss:", loss2)
    
    m = v_precise_grad(uset, numbers, s, dots, axis, omegas, Z, M, L, mic)
    v = np.square(m)
    b1 = 0.9
    b2 = 0.999
    big = np.arange(100000)
    small = np.arange(step)
    t = 0
    if stoch == 0:
        for k in big:
            for kk in small:
                grad = v_precise_grad(uset, numbers, s, dots, axis, omegas, Z, M, L, mic)
                m = b1*m+(1-b1)*grad
                v = b2*v+(1-b2)*np.square(grad)
                mtilde = m/(1-b1**(1+t))
                vtilde = v/(1-b2**(1+t))
                v_sqrt = np.sqrt(vtilde) + np.ones(len(grad))*1e-8
                uset = uset - learning_rate*np.divide(mtilde,v_sqrt)
                t+=1
            loss2 = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L, mic)
            print("Iteration:", t, "Loss function:", loss2)
            print("uset =", list(uset),"\n")
            if loss2 <= eps:
                break
    else:
        for k in big:
            for kk in small:
                grad = v_stoch_grad(uset, stoch, numbers, s, dots, axis, omegas, Z, M, L, mic)
                m = b1*m+(1-b1)*grad
                v = b2*v+(1-b2)*np.square(grad)
                mtilde = m/(1-b1**(1+t))
                vtilde = v/(1-b2**(1+t))
                v_sqrt = np.sqrt(vtilde) + np.ones(len(grad))*1e-8
                uset = uset - learning_rate*np.divide(mtilde,v_sqrt)
                t+=1
            loss2 = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L, mic)
            print("Iteration:", t, "Loss function:", loss2)
            print("uset =", list(uset),"\n")
            if loss2 <= eps:
                break
    
    return uset
