from pylion.lammps import lammps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from electrode import (System, PolygonPixelElectrode, PointPixelElectrode)
from scipy.optimize import (minimize, curve_fit)
import scipy.constants as ct
import gdspy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
import warnings
from skopt import dummy_minimize

"""
Functions, simulating the ion dynamics above planar traps
"""

@lammps.fix
def polygon_trap(uid, Omega, rf_voltages, dc_voltages, RFs, DCs, cover=(0, 0)):
    """
    Simulates an arbitrary planar trap with polygonal electrodes. Everything must be in standard units.

    Parameters
    ----------
    uid : str
        ions ID, .self parameter
    Omega : float or list shape([len(RF electrodes)])
        array of RF frequencies (or a frequency if they are the same) of each RF electrode
    rf_voltages : array shape([len(RF electrodes)])
        Set of the peak voltages on RF electrodes.
    dc_voltages : array shape([len(DC electrodes)])
        Set of the voltages on DC electrodes.
    RFs : list shape([len(RF electrodes), 4])
        Array of coordinates of RF electrodes in m
    DCs : list shape([len(DC electrodes), 4])
        Array of coordinates of DC electrodes in m
        The order of electrodes must match the order of voltage set and omegas.
    cover : list shape([2]), optional, default is (0,0)
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
    Simulates an arbitrary point trap. The point trap means a trap of an arbitrary shape, which is approximated by 
    circle-shaped electrodes, called points. The points potential is approximated from the fact, that it has 
    infinitesemal radius, so the smaller are points, the more precise is the simulation (but slower).
    
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
    cover : [cover_max, cover_height], optional, default is (0,0)
        array [cover_number, cover_height] - number of the terms incover electrode influence expansion 
        (5 is mostly enough) and its height.
    
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
    resolution : int, optional, default is 100
        number of points in the ring design
    cover : [cover_max, cover_height], optional, default is (0,0)
        array [cover_number, cover_height] - number of the terms in cover electrode influence expansion 
        (5 is mostly enough) and its height.
    Returns
    -------
    odict : str
        updates simulation of a single ring

    """
    _, trap = ring_trap_design(RF_voltage, Omega, r, R, r_dc, dc_voltage, res=resolution, need_coordinates=True, 
                               cheight=cover[0], cmax=cover[1])

    return point_trap(trap, cover)


"""
Shuttling of ions in the trap
"""

def linear_shuttling_voltage(s, x0, d, T, dc_set, shuttlers=0, N=4, vmin=-15, vmax=15, res=25, L=1e-6, need_func=False, 
                             freq_coeff=0, freq_ax=[0,1,2]):
    """
    Performs optimization of voltage sequence on DC electrodes for the linear shuttling, according to the tanh route. 
    Voltage is optimized to maintain axial secular frequency along the route.

    Parameters
    ----------
    s : electrode.System object
        Surface trap from electrode package
    x0 : array shape([3])
        Starting point of the shuttling
    d : float
        distance of the shuttling
    T : float
        Time of the shuttling operation
    dc_set : list shape([len(DC electrodes)])
        list of starting DC voltages
    shuttlers : list, optional, default is 0
        list of DC electrodes (defined as numbers), used for shuttling
        if ==0 then all dc electrodes participate in shuttling
        if ==[2,4,6] for example, only 2, 4 and 6th electrodes are participating,
        the rest are stationary dc
    N : int, optional, default is 4
        parameter in the tanh route definition (see q_tan)
    vmin : float, optional, default is -15
        Minimal allowable DC voltage.
    vmax : float, optional, default is 15
        Maximal allowable DC voltage. 
    res : int, optional, default is 25
        Number of steps of the voltage sequence during shuttling time.
    L : float, optional, default is 1e-6
        Dimension scale of the electrode. The default is 1e-6 which means um.
    need_func : bool, optional, default is False
        if True, the approximation functions of voltage sequences are provided, which are used for MD simulation of shuttling
    freq_coeff : float, optional, default is 0
        If not 0, optimimization will try to minimize secular frequency variations during shuttling. Coefficient 
        defines the input of the frequency variation to the loss function. If it's too high, optimization will not 
        succeed for shuttling, if it's too low, frequency variations will be ignored. Usually optimal value is between
        0.1 and 100, and has to be found manually.
    freq_ax : list [len up to 3], optional, default is (0,1,2)
        List of coordinate axes, along which the frequency variations are minimized, specified by index (0, 1, 2).
        For a single axis it must be, for example [0]. Default is [0,1,2], meaning all three coordinate axes.

    Returns
    -------
    voltage_seq : list shape([len(shuttlers), res+1])
        voltage sequences on each shuttler electrode (could be constant)
    funcs : list of strings shape([len(DC electrodes)]), optional
        list of functions in format, used for MD simulation (including stationary DC)

    """
    def q_tan(t):
        return np.array([d/2*(np.tanh(N*(2*t-T)/T)/np.tanh(N) + 1), 0, 0])    
    
    return shuttling_voltage(s, x0, q_tan, T, dc_set, shuttlers, vmin, vmax, res, L, need_func, freq_coeff, freq_ax)
    
def fitter_tan(t, a, b, c, d):
    return a*np.tanh(b*(t + c)) + d

def fitter_norm(t, a, b, c, d):
    return a*np.exp(b*(t-c)**2) + d

def approx_linear_shuttling(voltage_seq, T, res):
    """
    Approximation of voltage sequences on DC electrodes with analytic functions

    Parameters
    ----------
    voltage_seq : list shape([len(DC electrodes), res+1])
        voltage sequences on each DC electrode (could be constant)
    T : float
        Time of shuttling operation
    res : int
        Resolution of the voltage sequence definition.

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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt_tan, _ = curve_fit(fitter_tan, x_data, seq, [np.abs(dif/2), dif/T, -T/2, mean])
                    tan = np.linalg.norm(seq - fitter_tan(x_data, *popt_tan))
            except:
                att += 1
                tan = 1e6
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt_norm, _ = curve_fit(fitter_norm, x_data, seq, [np.abs(ampl), -1/2/T, T/2, np.min(seq)])
                    norm = np.linalg.norm(seq - fitter_norm(x_data, *popt_norm))
            except:
                att +=1
                norm = 1e6
            if att == 2:
                warnings.warn(f"Failed to fit {i}th electrode. Needs custom curve fitting")
            
            if tan > norm:
                funcs.append('((%5.6f) * exp((%5.6f) * (step*dt - (%5.6f))^2) + (%5.6f))' % tuple(popt_norm))
            else:
                funcs.append('((%5.6f) * (1 - 2/(exp(2*((%5.6f) * (step*dt + (%5.6f)))) + 1)) + (%5.6f))' % tuple(popt_tan))
                
    return funcs

def lossf_shuttle(uset, s, omegas, positions, L, dc_set, shuttlers, freq_coeff, freq_ax):
    '''
    Loss function for shuttling optimization

    Parameters
    ----------
    uset : np.array shape([len(shuttlers)])
        Varying voltage set on shuttling electrodes
    s : electrode.System object
        Surface trap from electrode package
    omegas : np.array shape([len(positions), len(freq_ax)])
        List of secular frequencies for each considered potential minimum at the begining of operation
    positions : np.array shape([len(potential minimums), 3])
        List of potential minimums
    L : float
        Dimension scale of the electrode. The default is 1e-6 which means um.
    dc_set : list shape([len(DC electrodes)])
        initial voltages on all dc electrodes
    shuttlers : list
        list of DC electrodes (defined as numbers), used for shuttling
        if ==0 then all dc electrodes participate in shuttling
        if ==[2,4,6] for example, only 2, 4 and 6th electrodes are participating,
        the rest are stationary dc
    freq_coeff : float
        If not 0, optimimization will try to minimize secular frequency variations during shuttling. Coefficient 
        defines the input of the frequency variation to the loss function. If it's too high, optimization will not 
        succeed for shuttling, if it's too low, frequency variations will be ignored. Usually optimal value is between
        0.1 and 100, and has to be found manually.
    freq_ax : list [len up to 3]
        List of coordinate axes, along which the frequency variations are minimized, specified by index (0, 1, 2).
        For a single axis it must be, for example [0]. Default is [0,1,2], meaning all three coordinate axes.

    Returns
    -------
    loss : float
        Quadratic norm of difference between current and desired minimum position. Scaled with norm of difference 
        between current secular frequency and desired one.
    '''
    rfss = np.array([0 for el in s if el.rf])
    for c, elec in enumerate(shuttlers):
        dc_set[elec] = uset[c]
    u_set = np.concatenate([rfss, np.array(dc_set)])
    loss = 0

    with s.with_voltages(dcs = u_set, rfs = None):
        if freq_coeff > 0:
            for i,x in enumerate(positions):
                try:
                    xreal = s.minimum(x*1.001, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG",
                                    tol = 1e-9, options = {'maxiter' : 100000})
                    curv_z, _ = s.modes(xreal,sorted=False)
                    curv_z = curv_z[[freq_ax]]
                except:
                    xreal = x * 1.1
                    curv_z = omegas[i] * 1.1
                loss += np.sqrt(np.linalg.norm(xreal - x) ** 2 + freq_coeff*L**(-2)*np.linalg.norm(curv_z - omegas[i])**2)
        else:
            for i,x in enumerate(positions):
                try:
                    xreal = s.minimum(x*1.001, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG",
                                    tol = 1e-9, options = {'maxiter' : 100000})
                except:
                    xreal = x * 1.1
                loss += np.linalg.norm(xreal - x)
            
    return loss ** 2


def shuttling_voltage(s, starts, routes, T, dc_set, shuttlers=0, vmin=-15, vmax=15, res=50, L=1e-6, need_func=False, 
                      freq_coeff=0, freq_ax=(0,1,2)):
    '''
    General function for shuttling voltage sequence, which optimizes the voltage sequence on chosen electrode for the 
    (arbitrary 3D) route of 1 ion or simulataneous varying routes of several different ions (potential wells)

    Parameters
    ----------
    s : electrode.System object
        Surface trap from electrode package
    starts : list shape([number of wells, 3])
        Array of initial coordinates of each potential well to be considered. May be only one point with shape([3])
    routes : callable, function(i:int, t:float/np.array)
        function determining the routes of potential wells in time. Must be in the following form: if only 1 well:
            def routes(t):
                return np.array([x(t), y(t), z(t)])
        if 2 or more wells shuttled simultaneously:
            def routes(i,t):
                if i == 0:
                    return np.array([x_0(t), y_0(t), z_0(t)])
                ...
                elif i == j:
                    return np.array([x_j(t), y_j(t), z_j(t)])          
    T : float
        time of shuttling operation
    dc_set : list shape([len(DC electrodes)])
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
    res : int, optional, default is 50
        Number of steps of the voltage sequence during shuttling time.
    L : float, optional, default is 1e-6
        Dimension scale of the electrode. The default is 1e-6 which means um.
    need_func : bool, optional, default is False
        if True, the approximation functions of voltage sequences on all DC electrodes
        are provided, which are used for MD simulation of shuttling
    freq_coeff : float, optional, default is 0
        If not 0, optimimization will try to minimize secular frequency variations during shuttling. Coefficient 
        defines the input of the frequency variation to the loss function. If it's too high, optimization will not 
        succeed for shuttling, if it's too low, frequency variations will be ignored. Usually optimal value is between
        0.1 and 100, and has to be found manually.
    freq_ax : list [len up to 3], optional, default is (0,1,2)
        List of coordinate axes, along which the frequency variations are minimized, specified by index (0, 1, 2).
        For a single axis it must be, for example [0]. Default is [0,1,2], meaning all three coordinate axes.

    Returns
    -------
    voltage_seq : list shape([len(shuttlers), res+1])
        voltage sequences on each shuttler electrode (could be constant)
    funcs : list of strings shape([len(DC electrodes)]), optional
        list of functions in format, used for MD simulation (including stationary DC)

    '''
    voltage_seq = []
    try:
        shuttlers[0]
    except:
        shuttlers = range(len(dc_set))
    try:
        starts[0][0]
    except:
        starts = np.array([starts])

    bnds = [(vmin,vmax) for el in shuttlers]
    rfss = np.array([0 for el in s if el.rf])
    u_set = np.concatenate([rfss, np.array(dc_set)])
    wells = len(starts)
    curves = []

    for i,start in enumerate(starts):
        with s.with_voltages(dcs = u_set, rfs = None):
            x1 = s.minimum(np.array(start), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
            curv_z, mod_dir=s.modes(x1,sorted=False)
            curvv = []
            for ax in freq_ax:
                curvv.append(curv_z[ax])
            curves.append(np.array(curvv))
            
    uset = np.zeros(len(shuttlers))
    for c, elec in enumerate(shuttlers):
        uset[c] = dc_set[elec]
    x = np.zeros((wells, 3))

    try:
        routes(0,0)
        multiple = True
    except:
        multiple = False

    if multiple:
        for dt in tqdm(range(res+1), desc='Shuttling optimization'):
            t = dt*T/res
            for i,x0 in enumerate(starts):
                x[i] = x0 + routes(i,t)
                
            newset = minimize(lossf_shuttle, uset, args = (s, curves, x, L, dc_set, shuttlers, freq_coeff, freq_ax), 
                              tol = 1e-6, bounds = bnds, options = {'maxiter' : 10000})
            uset = newset.x
            voltage_seq.append(uset) 
            
    else:
        for dt in tqdm(range(res+1), desc='Shuttling optimization'):
            t = dt*T/res
            for i,x0 in enumerate(starts):
                x[i] = x0 + routes(t)
                
            newset = minimize(lossf_shuttle, uset, args = (s, curves, x, L, dc_set, shuttlers, freq_coeff, freq_ax), 
                              tol = 1e-6, bounds = bnds, options = {'maxiter' : 10000})
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
        funcs = approx_linear_shuttling(volt_seq, T, res)
        return volt_seq, funcs
    else:
        return volt_seq

@lammps.fix
def polygon_shuttling(uid, Omega, rf_set, RFs, DCs, shuttlers, cover=(0, 0)):
    """
    Simulates an arbitrary planar trap with polygonal electrodes and shuttling in this trap. The shuttling is specified
    by the voltage sequence on the DC electrodes, in terms of some smooth function V(t).
    Any V(t) can be applied to this function. It must be specifyed as a string.
    For example: V(t) = np.exp(-k*t**2) == "exp(-k*(step*dt)^2)",
    If const: V(t) = V == "V"
    
    Parameters
    ----------
    uid : str
        ions ID, .self parameter
    Omega : float or list shape([len(RF electrodes])
        array of RF frequencies (or a frequency if they are the same) of each RF electrode
    rf_set : array shape([len(RF electrodes)])
        Set of the peak voltages on RF electrodes.
    RFs : list shape([len(RF electrodes), 4])
        Array of coordinates of RF electrodes in m
    DCs : list shape([len(DC electrodes), 4])
        Array of coordinates of DC electrodes in m
        The order of electrodes must match the order of voltage set and omegas.
    shuttlers : list of strings shape([len(DC electrodes)])
        list of strings, representing functions at which voltage on DC electrodes is applied through simulation time
    cover : list shape([2]), optional, default is (0,0)
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
    Function to almost optimally approximate the given shape (specified by boundary) with circles for a given resolution.
     Works as follows: the area of scale*scale size is packed with circles by hexagonal packing with resolution res. 
     Then, for nth electrode the shape, specified by boundary() is cutted from the area.

    Parameters
    ----------
    scale : float
        size of packed area
    boundary : callable, function(n:int, x:list of in plane coordinates [x,y])
        boundary function for a given electrode's shape. Returns True, is the point is within boundaries, or False, if not.
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


def n_wire_trap_design(top_dc, bottom_dc, central_wires, rf_indxs, Urf=0, gap=0, wire_splitting={}, L=1e-6, 
                     cheight=0, cmax=0, need_coordinates=False, need_plot=False, save_plot=None, figsize=(20,20)):
    '''
    Initializes the n-wire linear ion trap as *electrode* object. Each dc and rf electrode is labeled by its index, 
    matching with the voltages, applied to them. Consists of side DC electrodes and central wires, which can be either 
    DC or RF. Each wire can be separated into individual electrodes.

    Parameters
    ----------
    top_dc : array shape([number of top DC electrodes, 2])
        list of [x-width, y-width] of rectangular electrodes on top of the central wires. Will be placed from left to right.
    bottom_dc : array shape([number of bottom DC electrodes, 2])
        list of [x-width, y-width] of rectangular electrodes bottom of the central wires. Will be placed from left to right.
    central_wires : array shape([number of central DC electrodes, 2])
        list of [x-width, y-width] of rectangular central wires. Will be placed from top to bottom. 
    rf_indxs : list of shape([number of RF electrodes])
        list of indexes, corresponding to the central wires which are RF electrodes. If contains the index of splitted 
        wire, all splitted electrodes will be RF. Indexes start with 0, ordered from top to bottom.
    Urf : float or array of floats shape([number of RF electrodes]), optional, default is 0
        Scaled RF amplitude of the trap. If scaled from SU rf amplitude (Vrf) as Urf = Vrf*np.sqrt(Z/mass)/(2*L*Omega), 
        then accounts for pseudopotential approximation. Single Urf is all RF electrodes share the same RF drive, or an 
        array of individual Urf for each electrode.
    gap : float, optional, default is 0
        Uniform gap between the electrodes. If >0, all electrode sizes will be increased by the same gap for accurate 
        simulation. The same gap, applied to *polygons_to_gds* will produce the drawing with the desired gaps.
    wire_splitting : dict, optional, default is {}
        dictionary of the following structure: 
        {index of wire(int) : [[x-widths of electrodes(float)], [rf_indxs(int, could be empty)]]}
        separates the specified wire into electrodes of the specified length. The y-width of each such electrode is 
        the same, as in the original wire. rf_indxs specify which of these electrodes are rf. 
    L : float, optional, default is 1e-6
        Dimension scale of the electrode. The default is 1e-6 which means um.
    cheight : float, optional, default is 0
        Height of cover electrode - grounded plane above the trap.
    cmax : int, optional, default is 0
        Expansion order for cover electrode. Usually 5 is sufficient for correct calculations.
    need_coordinates : bool, optional, default if False
        If True, returns the coordinates, scaled by L to SU form, used by polygon_simulation(). 
    need_plot : bool, optional, default is False
        If True, returns a plot of the trap with specified RF electrode. The default is False. 
    save_plot : str, optional, default is None
        If not None, saves the plot under the provided filename.
    figsize : tuple, optional, default is (20,20)
        Optional size of the plot. If the electrode labels are too big, try increasing this size. 
        For more detailed control of the plot reproduce the respective code in the main.
    
    Returns
    -------
    s : electrode.System object
        Surface trap from *electrode*.
    RF_electrodes : list shape([number of RF electrodes, electrode shape]), optional
        returns RF electrodes coordinates in SU for MD simulation.
    DC_electrodes : list shape([number of DC electrodes, electrode shape]), optional
        returns DC electrodes coordinates in SU for MD simulation.

    '''
    #uniform format
    try:
        Urf[0]
    except:
        Urf = np.ones(len(rf_indxs))*Urf

    for j, wire in enumerate(central_wires):
        central_wires[j] = np.array(wire) + np.array([gap, gap])
    wh_len_y = np.sum(central_wires, axis = 0)[1]
    
    RF = []
    DC = []
    count_rf = 0
    count_dc = 0
    electrodes = []
    if 0 in wire_splitting: 
        if 0 in rf_indxs:
            wire_splitting[0][1] = list(np.arange(len(wire_splitting[0][0])))
        celectrode_coordinates = [[[-central_wires[0][0]/2, wh_len_y/2], 
                                   [-central_wires[0][0]/2, wh_len_y/2 - central_wires[0][1]], 
                                   [-central_wires[0][0]/2 + wire_splitting[0][0][0], wh_len_y/2 - central_wires[0][1]], 
                                   [-central_wires[0][0]/2 + wire_splitting[0][0][0], wh_len_y/2]]]
        if 0 in wire_splitting[0][1]:
            st = f'rf[{count_rf}]'
            count_rf += 1
            electrodes.append([st, [celectrode_coordinates[-1]]])
            RF.append(np.array(celectrode_coordinates[-1])*L)
        else:
            st = f'dc[{count_dc}]'
            count_dc += 1
            electrodes.append([st, [celectrode_coordinates[-1]]])
            DC.append(np.array(celectrode_coordinates[-1])*L)
            
        for k, width in enumerate(wire_splitting[0][0]):
            if k > 0:
                celectrode_coordinates.append([[celectrode_coordinates[k-1][3][0], wh_len_y/2], 
                                               [celectrode_coordinates[k-1][3][0], wh_len_y/2 - central_wires[0][1]], 
                                               [celectrode_coordinates[k-1][3][0] + wire_splitting[0][0][k],
                                                wh_len_y/2 - central_wires[0][1]], 
                                               [celectrode_coordinates[k-1][3][0] + wire_splitting[0][0][k], wh_len_y/2]])
                if k in wire_splitting[0][1]:
                    st = f'rf[{count_rf}]'
                    count_rf += 1
                    electrodes.append([st, [celectrode_coordinates[-1]]])
                    RF.append(np.array(celectrode_coordinates[-1])*L)
                else:
                    st = f'dc[{count_dc}]'
                    count_dc += 1
                    electrodes.append([st, [celectrode_coordinates[-1]]])
                    DC.append(np.array(celectrode_coordinates[-1])*L)
                    
    else:
        celectrode_coordinates = [[[-central_wires[0][0]/2, wh_len_y/2], 
                                   [-central_wires[0][0]/2, wh_len_y/2 - central_wires[0][1]], 
                                   [central_wires[0][0]/2, wh_len_y/2 - central_wires[0][1]], 
                                   [central_wires[0][0]/2, wh_len_y/2]]]
        if 0 in rf_indxs:
            st = f'rf[{count_rf}]'
            count_rf += 1
            electrodes.append([st, [celectrode_coordinates[-1]]])
            RF.append(np.array(celectrode_coordinates[-1])*L)
        else:
            st = f'dc[{count_dc}]'
            count_dc += 1
            electrodes.append([st, [celectrode_coordinates[-1]]])
            DC.append(np.array(celectrode_coordinates[-1])*L)
    for j, wire in enumerate(central_wires):
        if j > 0:
            if j in wire_splitting:
                if j in rf_indxs:
                    wire_splitting[j][1] = list(np.arange(len(wire_splitting[j][0])))
                for k, width in enumerate(wire_splitting[j][0]):
                    if k == 0:
                        celectrode_coordinates.append([[-wire[0]/2, celectrode_coordinates[-1][1][1]], 
                                                       [-wire[0]/2, celectrode_coordinates[-1][1][1] - wire[1]], 
                                                       [-wire[0]/2 + wire_splitting[j][0][0], 
                                                        celectrode_coordinates[-1][1][1] - wire[1]], 
                                                       [-wire[0]/2 + wire_splitting[j][0][0], celectrode_coordinates[-1][1][1]]])
                        if k in wire_splitting[j][1]:
                            st = f'rf[{count_rf}]'
                            count_rf += 1
                            electrodes.append([st, [celectrode_coordinates[-1]]])
                            RF.append(np.array(celectrode_coordinates[-1])*L)
                        else:
                            st = f'dc[{count_dc}]'
                            count_dc += 1
                            electrodes.append([st, [celectrode_coordinates[-1]]])
                            DC.append(np.array(celectrode_coordinates[-1])*L)
                    if k > 0:
                        celectrode_coordinates.append([[celectrode_coordinates[-1][3][0], celectrode_coordinates[-1][3][1]], 
                                                       [celectrode_coordinates[-1][3][0], celectrode_coordinates[-1][3][1] - wire[1]], 
                                                       [celectrode_coordinates[-1][3][0] + wire_splitting[j][0][k], 
                                                        celectrode_coordinates[-1][3][1] - wire[1]], 
                                                       [celectrode_coordinates[-1][3][0] + wire_splitting[j][0][k], 
                                                        celectrode_coordinates[-1][3][1]]])
                        if k in wire_splitting[j][1]:
                            st = f'rf[{count_rf}]'
                            count_rf += 1
                            electrodes.append([st, [celectrode_coordinates[-1]]])
                            RF.append(np.array(celectrode_coordinates[-1])*L)
                        else:
                            st = f'dc[{count_dc}]'
                            count_dc += 1
                            electrodes.append([st, [celectrode_coordinates[-1]]])
                            DC.append(np.array(celectrode_coordinates[-1])*L)
            else:
                celectrode_coordinates.append([[-wire[0]/2, celectrode_coordinates[-1][1][1]], 
                                              [-wire[0]/2, celectrode_coordinates[-1][1][1] - wire[1]], 
                                              [wire[0]/2, celectrode_coordinates[-1][1][1] - wire[1]], 
                                              [wire[0]/2, celectrode_coordinates[-1][1][1]]])
                if j in rf_indxs:
                    st = f'rf[{count_rf}]'
                    count_rf += 1
                    electrodes.append([st, [celectrode_coordinates[-1]]])
                    RF.append(np.array(celectrode_coordinates[-1])*L)
                else:
                    st = f'dc[{count_dc}]'
                    count_dc += 1
                    electrodes.append([st, [celectrode_coordinates[-1]]])
                    DC.append(np.array(celectrode_coordinates[-1])*L)
        
    RF = np.array(RF)
    DCc = np.array(DC)

    t_len = np.sum(top_dc, axis=0)[0]
    electrode_coordinates = [[[-t_len/2, wh_len_y/2], 
                              [-t_len/2 + top_dc[0][0], wh_len_y/2],
                              [-t_len/2 + top_dc[0][0], wh_len_y/2 + top_dc[0][1]], 
                              [-t_len/2, wh_len_y/2 + top_dc[0][1]]]]
    for j, elec in enumerate(top_dc):
        if j > 0:
            electrode_coordinates.append([[electrode_coordinates[j-1][1][0], electrode_coordinates[j-1][1][1]], 
                                          [electrode_coordinates[j-1][1][0] + elec[0], electrode_coordinates[j-1][1][1]], 
                                          [electrode_coordinates[j-1][1][0] + elec[0], electrode_coordinates[j-1][1][1] + elec[1]], 
                                          [electrode_coordinates[j-1][1][0], electrode_coordinates[j-1][1][1] + elec[1]]])
        st = f'dc[{count_dc}]'
        count_dc += 1
        electrodes.append([st, [electrode_coordinates[-1]]])
    electrode_coordinates = np.array(electrode_coordinates)*L
    DC = electrode_coordinates

    b_len = np.sum(bottom_dc, axis=0)[0]
    electrode_coordinates = [[[-b_len/2, -wh_len_y/2], 
                              [-b_len/2, -wh_len_y/2 - bottom_dc[0][1]],
                              [-b_len/2 + bottom_dc[0][0], -wh_len_y/2 - bottom_dc[0][1]], 
                              [-b_len/2 + bottom_dc[0][0], -wh_len_y/2]]]
    for j, elec in enumerate(bottom_dc):
        if j > 0:
            electrode_coordinates.append([[electrode_coordinates[j-1][3][0], electrode_coordinates[j-1][3][1]], 
                                          [electrode_coordinates[j-1][3][0], electrode_coordinates[j-1][3][1] - elec[1]], 
                                          [electrode_coordinates[j-1][3][0] + elec[0], electrode_coordinates[j-1][3][1] - elec[1]], 
                                          [electrode_coordinates[j-1][3][0] + elec[0], electrode_coordinates[j-1][3][1]]])
        st = f'dc[{count_dc}]'
        count_dc += 1
        electrodes.append([st, [electrode_coordinates[-1]]])
    electrode_coordinates = np.array(electrode_coordinates)*L
    DC = np.concatenate([DC, electrode_coordinates])
    DC = np.concatenate([DCc, DC])

    electrodes_n = []
    for i in range(count_rf):
        st = f'rf[{i}]'
        for el in electrodes:
            if el[0] == st:
                electrodes_n.append(el)
    
    for i in range(count_dc):
        st = f'dc[{i}]'
        for el in electrodes:
            if el[0] == st:
                electrodes_n.append(el)
            

    # Polygon approach. All DCs are 0 for now
    s = System([PolygonPixelElectrode(cover_height=cheight, cover_nmax=cmax, name=n, paths=map(np.array, p))
                for n, p in electrodes_n])

    if len(Urf) != count_rf:
        warnings.warn('Provided RF voltages don\'t match RF electrodes')
    
    for i in range(count_rf):
        s[f'rf[{i}]'].rf = Urf[i]

    # creates a plot of electrode
    if need_plot:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        s.plot(ax[0])
        s.plot_voltages(ax[1], u=s.rfs)
        # u = s.rfs sets the voltage-type for the voltage plot to RF-voltages (DC are not shown)
        xmax = np.max(central_wires, axis = 0)[0]/2*1.2
        ymaxp = (np.max(top_dc, axis=0)[1] + wh_len_y/2) * 1.2
        ymaxn = (np.max(bottom_dc, axis=0)[1] + wh_len_y/2) * 1.2
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
        return s, RF, DC
    else:
        return s

def five_wire_trap_design(Urf, top_dc, bottom_dc, top_rf, bottom_rf, central_dc, gap=0, L=1e-6, 
                          cheight=0, cmax=0, need_coordinates=False, need_plot=False, save_plot=None, figsize=(20,20)):
    '''
    Subroutine to initialize simple five-wire trap as *electrode* object.

    Parameters
    ----------
    Urf : float
        Scaled RF amplitude of the trap. If scaled from SU rf amplitude (Vrf) as Urf = Vrf*np.sqrt(Z/mass)/(2*L*Omega), 
        then accounts for pseudopotential approximation.
    top_dc : array shape([number of top DC electrodes, 2])
        list of [x-width, y-width] of rectangular electrodes on top of the central wires. Will be placed from left to right.
    bottom_dc : array shape([number of bottom DC electrodes, 2])
        list of [x-width, y-width] of rectangular electrodes bottom of the central wires. Will be placed from left to right.
    top_rf : array shape([2])
        [x-width, y-width] of rectangular RF electrode on top.
    bottom_rf : array shape([2])
        [x-width, y-width] of rectangular RF electrode at bottom.
    central_dc : array shape([2])
        [x-width, y-width] of rectangular central DC electrode.
    *kwargs :
        Optional parameters from *n_wire_trap_design* exculding wire_splitting

    Returns
    -------
    s : electrode.System object
        Surface trap from *electrode*.
    RF_electrodes : list shape([number of RF electrodes, electrode shape]), optional
        returns RF electrodes coordinates in SU for MD simulation.
    DC_electrodes : list shape([number of DC electrodes, electrode shape]), optional
        returns DC electrodes coordinates in SU for MD simulation.
    '''
    rf_indxs = [0,2]
    central_wires = [top_rf, central_dc, bottom_rf]
    
    return n_wire_trap_design(top_dc, bottom_dc, central_wires, rf_indxs, Urf=[Urf, Urf], gap=gap, wire_splitting={}, 
                              L=L, cheight=cheight, cmax=cmax, need_coordinates=need_coordinates, need_plot=need_plot, 
                              save_plot=save_plot, figsize=figsize)


def ring_trap_design(Urf, Omega, r, R, r_dc=0, v_dc=0, res=100, need_coordinates=False, need_plot=False, cheight=0,  
                     cmax=0, save_plot=None, figsize=(13,5)):
    '''
    Function for designing the ring RF trap with (optionally) central DC electrode

    Parameters
    ----------
    Urf : float
        Scaled RF amplitude of the trap. If scaled from SU rf amplitude (Vrf) as Urf = Vrf*np.sqrt(Z/mass)/(2*L*Omega), 
        where Z - ion's charge in SU, then accounts for pseudopotential approximation.
    Omega : float
        RF frequency of the trap.
    r : float
        inner radius of the electrode ring in SU.
    R : float
        outter radius of the electrode ring in SU.
    r_dc : float, optional, default is 0
        radius of central DC circled electrode. 
    v_dc : float, optional, default is 0
        Voltage of dc electrode. 
    res : int, optional, default is 100
        Resolution of the trap.
    need_coordinates : bool, optional, default is False
        If True, returns "trap" object for point_trap() simulation.
    need_plot : bool, optional, default is False
        If True, returns a plot of the trap, with RF and DC electrode voltages separately.
    cheight : float, optional, default is 0
        Height of cover electrode - grounded plane above the trap. 
    cmax : int, optional, default is 0
        Expansion order for cover electrode. Usually 5 is sufficient for correct calculations. 
    save_plot : str, optional, default is None
        If not None, saves the plot under the provided filename.

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
    
    return point_trap_design(frequencies, rf_voltages, dc_voltages, ring_boundary, scale, res, need_coordinates=need_coordinates, 
                             need_plot=need_plot, cheight=cheight, cmax=cmax, save_plot=save_plot, figsize=figsize)

def point_trap_design(frequencies, rf_voltages, dc_voltages, boundaries, scale, resolution, need_coordinates=False, 
                      need_plot=False, cheight=0, cmax=0, save_plot=None, figsize = (13,5)):
    '''
    Function for designing arbitrarily shape point trap. Each electrode shape is determined by provided boundary.

    Parameters
    ----------
    frequencies : float or list shape([len(RF electrodes)])
        array of RF frequencies (or a frequency if they are the same) of each RF electrode
    rf_voltages : array shape([len(RF electrode)])
        Set of the peak voltages on RF electrodes.
    dc_voltages : array shape([len(DC electrodes]))
        Set of the voltages on DC electrodes.
    boundary : callable, function(n:int, x:list of in plane coordinates [x,y])
        boundary function for a given electrode's shape. Returns True, is the point is within boundaries, or False, if not.
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
    resolution : int, optional, default is 100
        Resolution of the trap.
    need_coordinates : bool, optional, default is False
        If True, returns "trap" object for point_trap() simulation.
    need_plot : bool, optional, default is False
        If True, returns a plot of the trap, with RF and DC electrode voltages separately.
    cheight : float, optional, default is 0
        Height of cover electrode - grounded plane above the trap. 
    cmax : int, optional, default is 0
        Expansion order for cover electrode. Usually 5 is sufficient for correct calculations. 
    save_plot : str, optional, default is None
        If not None, saves the plot under the provided filename.

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
            fig, ax = plt.subplots(1,2,figsize=figsize)
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
            fig, ax = plt.subplots(1,2,figsize=figsize)
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

def polygons_from_gds(gds_lib, L=1e-6, cheight=0, cmax=0, need_coordinates=True, need_plot=True, save_plot=None):
    '''
    Creates polygon trap from GDS file, consisting this trap. It then plots the trap with indexes for each electrode, 
    for convenient defining of all voltages.

    Parameters
    ----------
    gds_lib : file.GDS
        file with GDS structure. This function will read only the top layer.
    L : float, optional, default is 1e-6
        Dimension scale of the electrode. The default is 1e-6 which means um.
    cheight : float, optional, default is 0
        Height of cover electrode - grounded plane above the trap.
    cmax : int, optional, default is 0
        Expansion order for cover electrode. Usually 5 is sufficient for correct calculations.
    need_coordinates : bool, optional, default if False
        If True, returns the coordinates, scaled by L to SU form, used by polygon_simulation(). 
    need_plot : bool, optional, default is False
        If True, returns a plot of the trap with each electrode assigned its index along the order from the GDS file. 
    save_plot : str, optional, default is None
        If not None, saves the plot under the provided filename.

    Returns
    -------
    s : electrode.System object
        Surface trap from electrode.
    full_elec : list shape([number of electrodes, electrode shape]), optional
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
    

def polygons_reshape(full_electrode_list, order, L=1e-6, need_plot=True, need_coordinates=True, 
                     cheight=0, cmax=0, save_plot=None):
    '''
    Sometimes it is convenient to have specific order of electrodes: RF starting first and so on.
    This function will reorder the electrode array, obtained from polygon_to_gds(), for a desired order.

    Parameters
    ----------
    full_electrode_list : list shape([number of electrodes, electrode shape])
        electrodes coordinates in SU from polygon_to_gds().
    order : list shape([number of electrodes])
        Desired order, along which electrode indeces will be reassigned.
    L : float, optional, default is 1e-6
        Dimension scale of the electrode. The default is 1e-6 which means um.
    need_plot : bool, optional, default is False
        If True, returns a plot of the trap with each electrode assigned its index 
        along the order from the GDS file. 
    need_coordinates : bool, optional, default is False
        If True, returns the coordinates, scaled by L to SU form, used by polygon_simulation().
    cheight : float, optional, default is 0
        Height of cover electrode - grounded plane above the trap. 
    cmax : int, optional, default is 0
        Expansion order for cover electrode. Usually 5 is sufficient for correct calculations. 
    save_plot : str, optional, default is None
        If not None, saves the plot under the provided filename.

    Returns
    -------
    s : electrode.System object
        Surface trap from electrode .
    full_elec : list shape([number of electrodes, electrode shape]), optional
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
    elec : np.array shape([number of points in electrode, 2])
        Electrode to shrink, defined by its coordinates.
    gap : float
        Length to shrink the electrode. Corresponds to the width of the gap /2.

    Returns
    -------
    gapped : np.array shape([number of points in electrode, 2])
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

def polygon_to_gds(trap, name, gap=0):
    '''
    Creates GDS file of the trap with a given (uniformal) gap between electrodes

    Parameters
    ----------
    trap : list shape([number of electrodes, electrode shape])
        List of electrode coordinates for each electrode in the trap. Must be in um scale.
    name : str
        Name of the file to create GDS. Must end with '.gds'.
    gap : float, optional, default is 0
        Gap width between two electrodes. 

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
    x : np.array shape([3])
        (x, y, z) coordinates of a central point of ion cloud.
    number : int
        Number of ions.
    radius : float
        Radius of a cloud.

    Returns
    -------
    positions : np.array shape([number, 3])
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
    x : np.array shape([3])
        (x, y, z) coordinates of a starting point of ion chain.
    number : int
        Number of ions.
    dist : float
        distance between ions in the chain.

    Returns
    -------
    positions : np.array shape([number, 3])
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
    !!! This function is copied from the electrode package without changes.
     It is required here to avoid deprecation with numpy > 1.20. !!!

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
    Small *electrode* wrapper for more convenient trap frequency calculation. Calculates either secular modes of
    mathieu modes for a single ion.

    Parameters
    ----------
    s : electrode.System object
        Surface trap from *electrode*.
    potential_minimum : np.array shape([3])
        Potential minimum position for this trap.
    ion_mass : float
        Ion mass in standard units.
    L : float, optional, default is 1e-6
        Dimension scale of the electrode. The default is 1e-6 which means um.
    charge : int, optional, default is 1
        Ion charged, scaled to elementary charge.
    mathieu : bool, optional, default if False
        If True -- calculates Mathieu modes, if False -- calculates secular modes.
    r : int, optional, default is 2
        Band cutoff for Mathieu modes calculation.
    Omega : float, optional, default is None
        If not None -- RF frequency of the trap. Required for Mathieu modes calculation.

    Returns
    -------
    omegas : np.array shape([3])
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

def equilibrium_ion_positions(s, dc_set, ion_masses, ion_number, potential_minimum=None, L=1e-6, 
                              charges=1, positions_guess=None):
    '''
    Calculates the equilibrium ion positions in a crystal in pseudopotential approximation by minimizing 
    the system's energy.

    Parameters
    ----------
    s : electrode.System object
        Surface trap from electrode .
    dc_set : array shape([len(DC electrodes)])
        Set of the voltages on DC electrodes.
    ion_masses : np.array shape([ion number]) or float
        Either array of ion masses for mixed species crystal, or a single ion mass for a single species crystal
    ion_number : int
        Ion number
    potential_minimum : np.array shape([3]), optional, default is None
        If not None -- guess for the potential minimum position. Not needed, if positions_guess is not None.
    L : float, optional, default is 1e-6
        Dimension scale of the electrode. The default is 1e-6 which means um.
    charges : np.array shape([ion_number]) or int, optional, default is 1
        Array of ion charges for crystals with varying charged, or a single charge for uniformly charged ions. 
        Scaled for elementary charge. 
    positions_guess : np.array shape([ion_number, 3]), optional, default is None
        If not None -- initial guess for ion positions in the crystal. If None, potential_minimum should be not None.

    Returns
    -------
    equilibrium positions : np.array shape([ion_number, 3])
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
        x : np.array shape([3 * ion_number])
            concatentated array of ion coordinates

        Returns
        -------
        pot : float
            Potential energy in eV
        '''
        x = np.array_split(x, N)
        rf_set = s.rfs[s.rfs != 0]
        pot = 0
        for i, x0 in enumerate(x):
            with s.with_voltages(dcs = u_set, rfs = rf_set*np.sqrt(ion_masses[0]*charges[i]/ion_masses[i]/charges[0])):
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
    A hessian for N ion modes and 3 principle axes is defined in the following form:
        [[XX XY XZ]
         [YX YY YZ]
         [ZX ZY ZZ]], XX = matrix (N*N)

    Parameters
    ----------
    ion_positions : np.array shape([ion number, 3])
        Array of equilibrium ion positions.
    omega_sec : np.array shape([ion number, 3])
        Array of secular frequencies in 3 axes for each ion.
    ion_masses : np.array shape([ion number])
        Array of ion masses.
    charges : np.array shape([ion number])
        Array of ion charges.

    Returns
    -------
    A_matrix : np.array shape([3*ion_number, 3*ion_number])
        Hessian.
    M_matrix : np.array shape([3*ion_number, 3*ion_number])
        Mass matrix, used to retrieve normal modes.

    '''
    ion_positions = np.array(ion_positions)
    omega_sec = np.array(omega_sec)*2*np.pi
    l = (ct.e**2 / (4*np.pi*ct.epsilon_0*ion_masses[0]*(omega_sec[0][2])**2))**(1/3)
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
    Calculation of normal modes for an arbitrary configuration of ion crystal in harmonic potential. Arbitrary means 
    spatially linear or nonlinear (2D, 3D) crystals, of single or mixed-species ions, in individual potential 
    configurations (optical tweezers, individual micro traps), with arbitrary charges. This function works not only 
    for surface traps, but for any trap, as long as necessary parameters are provided.

    Parameters
    ----------
    ion_positions : np.array shape ([ion number, 3])
        Array of equilibrium ion positions.
    omega_sec : np.array shape([ion number, 3]) or ([3])
        Array of secular frequencies in 3 axes for each ion. Or a single 
        secular frequency, if they are equal for each ion
    ion_masses : np.array shape([ion number]) or float
        Array of ion masses. Or a single mass, if all the ions have equal mass.
    charges : np.array shape([ion_number]) or int, optional, default is 1
        Array of ion charges for crystals with varying charged, or a single charge for uniformly charged ions. 
        Scaled for elementary charge. 
    linear : bool, optional, default if False
        If True it will return only normal modes for each of principle axes of
        oscillation (x,y,z). correct for linear ion chains.
    reshape : bool, optional, default is True
        If True, result reshapes so that first come the modes, where maximal value
        of mode vector is in x direction (ie X modes), then Y modes, then Z modes. 

    Returns
    -------
    norm_freqs : np.array shape([3*ion number])
        Array of normal mode frequencies for each mode, in Hz (so scaled by 2pi).
    norm_modes : np.array shape([3*ion number, 3*ion_number])
        Normal mode matrix for every principle axis of oscillation. Each mode 
        has the following structure: the value of mode vector is given along x(y,z)-axis
        [x_axis[0],...x_axis[ion number], y_axis[0],...y_axis[ion number], z_axis[0],...z_axis[ion number]]
        Here the axes correspond to the directions of secular frequencies (of the first ion in crystal).
        
    Optional returns in case of linear chain (linear=True)
    ----------------------------------------
    norm_freqs : np.array shape([3, ion number])
        Array of normal mode frequencies for each principle axis (x,y,z),
        in Hz (so scaled by 2pi).
    norm_modes : np.array shape([3, ion number, ion_number])
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
    Returns Hessian of Coulomb potential for ions near equilibrium postions in harmonic approximation.

    Parameters
    ----------
    ion_positions : np.array shape([ion number, 3])
        Array of equilibrium ion positions.
    charges : np.array of ints shape([ion_number])
        Array of ion charges, scaled for elementary charge.

    Returns
    -------
    A_matrix : np.array shape([3*ion number, 3*ion number])
        Coulomb Hessian in SU.

    '''
    ion_positions = np.array(ion_positions)
    kap = ct.e**2 / (4*np.pi*ct.epsilon_0)
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
    ion_positions : np.array shape([ion number, 3])
        Array of equilibrium ion positions.
    s : electrode.System object
        Surface trap from electrode package
    rf_set : list shape([len(RF electrodes)])
        List of RF amplitudes for each electrode in SU.
    Omega : float
        RF frequency of the trap
    dc_set : list shape([len(DC electrodes)])
        Set of the voltages on DC electrodes.
    ion_masses : np.array shape([ion number])
        Array of ion masses.
    charges : np.array of ints shape([ion_number])
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
    A : np.array shape([3*ion number, 3*ion number])
        Anharmonic Hessian of the trap DC potential in SU.
    Q : np.array shape([3*ion number, 3*ion number])
        Anharmonic Hessian of the trap RF potential in SU.
    M_matrix : np.array([3*ion_number, 3*ion_number])
        Mass matrix, used to retrieve normal modes for single and mixed species crystals.

    '''
    u_set = np.concatenate((np.zeros(len(rf_set)), dc_set))
    rf_set = np.array(rf_set)
    A_hes = []
    Q_hes = []

    for i,pos in enumerate(ion_positions):
        Urf = rf_set*np.sqrt(ct.e*charges[i]/ion_masses[i])/(2*L*Omega)

        with s.with_voltages(dcs = u_set, rfs = Urf):
            min_pos = np.array(s.minimum( 1.001*pos/L, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG"))

        with s.with_voltages(dcs = u_set, rfs = rf_set):
            A_deriv2 = s.electrical_potential(min_pos, "dc", 2, expand=True)[0]*ct.e/L**2 
            Q_deriv2 = s.electrical_potential(min_pos, "rf", 2, expand=True)[0]*ct.e/L**2

            A_deriv3 = np.zeros([3,3])
            Q_deriv3 = np.zeros([3,3])
            pota = s.electrical_potential(min_pos, "dc", 3, expand=True)[0]*ct.e/L**2 
            potq = s.electrical_potential(min_pos, "rf", 3, expand=True)[0]*ct.e/L**2
            
            for r in range(N):
                for p in range(3):
                    A_deriv3 = A_deriv3 + 1/3*(ion_positions[r][p]/L-min_pos[p])*pota[p]
                    Q_deriv3 = Q_deriv3 + 1/3*(ion_positions[r][p]/L-min_pos[p])*potq[p]

            A_deriv4 = np.zeros([3,3])
            Q_deriv4 = np.zeros([3,3])
            potaa = s.electrical_potential(min_pos, "dc", 4, expand=True)[0]*ct.e/L**2
            potqq = s.electrical_potential(min_pos, "rf", 4, expand=True)[0]*ct.e/L**2

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

def crystal_modes(ion_positions, ion_masses, s, rf_set, Omega, dc_set, L = 1e-6, charges = 1, 
                  reshape=True, anharmonic=True):
    '''
    Returns harmonic modes of an arbitrary ion crystal configuration in surface trap, considering time-dependent 
    potential as in 3N dimensional Mathieu equation and hexapole and octopole anharmonic potential terms of the trap.

    Parameters
    ----------
    ion_positions : np.array shape([ion number, 3])
        Array of equilibrium ion positions.
    ion_masses : np.array shape([ion number]) or float
        Array of ion masses. Or a single mass, if all the ions have equal mass.
    s : electrode.System object
        Surface trap from *electrode* package
    rf_set : list shape([len(RF electrodes)])
        List of RF amplitudes for each electrode in SU.
    Omega : float
        RF frequency of the trap
    dc_set : array shape([len(DC electrodes)])
        Set of the voltages on DC electrodes.
    L : float, optional, default is 1e-6
        Dimension scale of the electrode. The default is 1e-6 which means um.
    charges : array of ints shape([ion_number]) or int, optional, default is 1
        Array of ion charges for crystals with varying charged, or a single charge for uniformly charged ions. 
        Scaled for elementary charge. 
    reshape : bool, optional. default is True
        If True, result reshapes so that first come the modes, where maximal value
        of mode vector is in x direction (ie X modes), then Y modes, then Z modes.
        !!Note may give unexpected results for anharmonic modes, which reflects anharmonic mixing of modes on different
        principle oscillation axes/ !!

    Returns
    -------
    norm_freqs : np.array shape([3*ion number])
        Array of harmonic mode frequencies for each mode, in Hz.
    norm_modes : np.array shape ([3*ion number, 3*ion_number])
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

def anharmonics(s, minimums, axis, L=1e-6):
    """
    Calculates anharmonic scale lengths at given coordinates. The lengthes are given for harmonic, n = 3 and
     n = 4 terms. Length are defined as in DOI 10.1088/1367-2630/13/7/073026 .

    Parameters
    ----------
    s : electrode.System object
        System object from electrode package, defining the trap
    minimums : array shape([number of positions, 3])
        Coordinates of approximate minimum positions (the exact mimums will be
                                                      calculated from them)
    axis : int
        axis, along which the anharmonicity is investigated. 
        Determined by respecting int: (0, 1, 2) = (x, y, z)
    L : float, optional, default is 1e-6
        Dimension scale of the electrode. The default is 1e-6 which means um.

    Returns
    -------
    results : list shape [number of positions, 3]
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
        
        k2 = deriv2[axis]*ct.e/L**2
        k3 = np.linalg.eig(deriv3)[0][axis]*ct.e
        k4 = np.linalg.eig(deriv4)[0][axis]*ct.e
        l2 = (ct.e**2/(4*np.pi*ct.epsilon_0*k2))**(1/3)
        l3 = (k3/k2)**(1/(2-3))
        l4 = np.sign(k4)*(np.abs(k4)/k2)**(1/(2-4))
        results.append([l2, l3, l4])
    
    return results


"""
Stability analysis
"""

def stability(s, ion_masses, Omega, minimum, charges=1, L=1e-6, need_plot=True, save_plot=None):
    """
    Returns stability parameters for the linear planar trap (RF confinement only radial) If asked, return plot of the 
    stability a-q diagram for this trap and plots the parameters for this voltage configuration (as a color dot) 
    for given isotopes.

    Parameters
    ----------
    s : electrode.System object
        System object from electrode package, defining the trap
    ion_masses : float or list shape([number of species])
        Either array of ion masses to analyse several species, or a single ion mass for a single species
    Omega : float
        RF frequency 
    minimum : list shape (3)
        approximate position of potential minimum, from which the real minimum is calculated
    charge : int or list of ints shape([number of species]), optional, default is 1
        Array of ion charges for species with varying charge, or a single charge for uniformly charged ion species. 
        Scaled for elementary charge. 
    L : float, optional, default is 1e-6
        Dimension scale of the electrode. The default is 1e-6 which means um.
    need_plot : int, optional, default is True
        if True, then the plot of the stability diagram and the a-q parameters for this voltage configuration is shown. 
    save_plot : str, optional, default is None
        If not None, saves the plot under the provided filename.

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
    try:
        ion_masses[0]
    except:
        ion_masses = [ion_masses]
    
    try:
        charges[0]
    except:
        charges = np.ones(len(ion_masses)) * charges
    
    charges = np.array(charges) * ct.e
    params = {}
    for M, Z in zip(ion_masses, charges):
        rf_set = s.rfs
        rf_set = rf_set*np.sqrt(Z/M)/(2*L*Omega)
        scale = Z/((L*Omega)**2*M)
        with s.with_voltages(dcs = None, rfs = rf_set):
            x1 = s.minimum(np.array(minimum), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")

            a = 4*scale*s.electrical_potential(x1, "dc", 2, expand=True)[0] 
            q = 4*np.sqrt(scale)*s.electrical_potential(x1, "rf", 2, expand=True)[0]      
            a, q = np.array(a), np.array(q)
            A = a
            Q = q
            A = A[1:3, 1:3]
            Q = Q[1:3, 1:3]

            DiagA, BasA = np.linalg.eig(A)
            DiagA = np.diag(DiagA)
            areal = DiagA[0,0]
            alpha = np.abs(-DiagA[1,1]/areal)

            DiagQ, BasQ = np.linalg.eig(Q)
            DiagQ = np.diag(DiagQ)
            qreal = DiagQ[0,0]
            params[f'Ion (M = {round(M/ct.atomic_mass):d}, Z = {round(Z/ct.e):d})'] = {'a':areal, 'q':qreal}

    rot = np.dot(np.linalg.inv(BasQ),BasA)
    thetha = np.arccos(rot[0,0])
    c = np.cos(2*thetha)
    sin = np.sin(2*thetha)
    params['\u03B1'] = alpha
    params['\u03B8'] = thetha

    q = np.linspace(0,1.5, 1000)
    ax = -q**2/2
    ab = q**2/(2*alpha)
    ac = 1 - c*q - (c**2/8 + (2*sin**2*(5+alpha))/((1+alpha)*(9+alpha)))*q**2
    ad = -(1 - c*q - (c**2/8 + (2*sin**2*(5+1/alpha))/((1+1/alpha)*(9+1/alpha)))*q**2)/alpha

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
        fig = plt.figure()
        fig.set_size_inches(7,5)
        plt.plot(q,ax, 'g')
        plt.plot(q,ab, 'g')
        plt.plot(q,ac, 'g')
        plt.plot(q,ad, 'g')
        
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
       
        colors=["maroon", 'peru',"darkgoldenrod",'magenta', "orangered", 'darkorange', 'crimson', 'brown']
        k = 0
        for M, Z in zip(ion_masses, charges):
            plt.scatter(params[f'Ion (M = {round(M/ct.atomic_mass):d}, Z = {round(Z/ct.e):d})']['q'], params[f'Ion (M = {round(M/ct.atomic_mass):d}, Z = {round(Z/ct.e):d})']['a'], s = 40, edgecolor='black', color = colors[k], label = f'Ion (M = {round(M/ct.atomic_mass):d}, Z = {round(Z/ct.e):d})' )
            k = (k+1)%8
        plt.legend()
        plt.tight_layout()
        if save_plot:
            plt.savefig(save_plot)
    
        plt.show()
        
    return params

"""
Voltage optimization functions
"""

def frequency_optimization(s, ion_masses, positions, axis, omegas, start_dcset, numbers=0, eps=1e-9, tol=1e-18,
                         find_initial_guess=False, call_num=2000, charges=1, L=1e-6, callback_num=100, micro=0, 
                         voltage_bounds=(-15,15), mathieu=False, r_mathieu=2, Omega_rf=None):
    """
    Optimizes set of voltages on specified electrodes, so that the secular frequencies in all potential minima match 
    the required frequency set. On request can try to keep potential minima in the same positions. Optimizes both 
    secular and mathieu modes, performs with mixed species crystals.

    Parameters
    ----------
    s : electrode.System object
        System object from *electrode* package, defining the trap
    ion_masses : np.array shape([ion number]) or float
        Either array of ion masses for mixed species crystal, or a single ion mass for a single species crystal
    positions : list shape([number of positions, 3])
        Positions of potential minima, at which ion frequencies are calculated
    axis : list shape([number of axes])
        List of all principle axes, at which the desired secular frequencies are determined. 
        Defined as (0, 1, 2) = (x, y, z). For example, if axis == (1), only frequencies at y axis are considered.
        If axis == (0, 1, 2), all axes are considered simultaneously
    omegas : list shape([len(axis), number of positions])
        Set of desired secular frequencies in Hz (/2pi), given for all positions and axes. 
    start_dcset : list shape([number of dc electrodes])
        Starting DC voltages for the optimization (on all electrodes).
    numbers : list shape([number of optimizing electrodes]) or int, optional, default is 0
        list of the indeces of DC electrodes, used for optimization. If 0, all DC elecrtodes are used for optimization.
    eps : float, optional, default is 1e-9
        Maximal loss, at which the algorithm considers the implementation succesful. If lower eps are possible, 
        the algorithm will keep optimizing according to tol.
    tol : float, optional, default is 1e-18
        Tolerance for termination from scipy.optimize.
    find_initial_guess : bool, optional, default is False
        If True, finds the best initial guess for the voltage optimization via randomized grid search. If False, will 
        just use start_dcset. 
    call_num : int, optional, default is 2000
        Number of grid search loss function calls.
    charges : array shape([ion_number]) or int, optional, default is 1
        Array of ion charges for crystals with varying charged, or a single charge for uniformly charged ions. 
        Scaled for elementary charge. 
    L : float, optional, default is 1e-6
        Dimension scale of the electrode. The default is 1e-6 which means um.
    callback_num : int, optional, default is 100
        Number of iterations, at which the current voltage set will be as callback printed.
    micro : float, optional, default is 0
        If non 0, algorithm will attempt to compensate micromotion by keeping potential minima at the starting positions.
        Coeff value coresponds to the input of minimum displacement in loss function, and should be evaluated emperically.
        Approximate range: 0.001 to 10
    voltage_bounds : tuple (float,float) or list of tuples len(number of dc electrodes), optional, default is (-15,15)
        List of voltage bounds on each DC electrode. If tuple, then each electrode will be assigned the same bounds. 
        Alternatively, each electrode can be assigned individual bounds as [(-bnds_1, bnds_1), (-bnds_2,bnds_2),...]
    mathieu : bool, optional, default is False
        If True -- optimizes Mathieu modes, if False -- optimizes secular modes.
    r_mathieu : int, optional, default is 2
        Band cutoff for Mathieu modes calculation.
    Omega : float, optional, default is None
        If not None -- RF frequency of the trap. Required for Mathieu modes calculation.

    Returns
    -------
    dcset : list shape([number of dc electrodes])
        Resulting DC voltage set.

    """
    positions = np.array(positions)
    try:
        numbers[0]
    except:
        numbers = np.arange(len(start_dcset))
    
    try:
        M = ion_masses[0]
    except:
        ion_masses = np.ones(positions.shape[0])*ion_masses
        M = ion_masses[0]
    try:
        Z = charges[0]
    except:
        charges = np.ones(positions.shape[0]) * charges
        Z = charges[0]
    
    rf_set = s.rfs
    numbers = np.array(numbers)
    omegas = np.array(omegas).T
    start_dcset = np.array(start_dcset)

    def v_lossf(uset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start_dcset[[numbers]] = uset
            pbarn.update(1)
            loss = 0
            u_set = np.zeros(len([rfs for rfs in s.rfs if rfs != 0]))
            u_set = np.concatenate([u_set, start_dcset])
            
            for i, pos in enumerate(positions):
                with s.with_voltages(dcs=u_set, rfs=rf_set*np.sqrt(M*charges[i]/ion_masses[i]/Z)):
                    try:
                        x1 = s.minimum(pos, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG",
                                    tol = 1e-5, options = {'maxiter' : 800})
                        omega_sec, _ = single_ion_modes(s, x1, ion_masses[i], L=L, charge=charges[i], 
                                                        mathieu=mathieu, r=r_mathieu, Omega=Omega_rf)
                        omega_sec = omega_sec[[axis]]

                        if np.any(np.isnan(omega_sec)):
                            loss += 100
                        else: 
                            loss += np.linalg.norm((omega_sec-omegas[i])/omegas[i][0])**2
                        
                        loss += micro*np.linalg.norm(x1-pos)**2
        
                    except: 
                        loss += 100

        return loss

    count=[0]
    def callback(intermediate_result):
        pbar.update(1)
        count[0] += 1
        if count[0] % callback_num == 0:
            print(f'Current loss = {intermediate_result.fun}')
            print(f'Current uset = {list(intermediate_result.x)}')
        return count

    try:
        voltage_bounds[0][0]
        voltage_bounds=list(voltage_bounds)
    except:
        voltage_bounds = [(voltage_bounds[0],voltage_bounds[1]) for n in range(len(numbers))]

    if find_initial_guess:
        with tqdm(total=call_num, desc='Finding initial guess') as pbarn:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = dummy_minimize(v_lossf, voltage_bounds, n_calls=call_num)
                initial_uset = res.x
                print(f'Initial loss = {res.fun}')
                    
    else:
        with tqdm(desc='Optimizing voltages', disable=True) as pbarn:
            uset = np.array(start_dcset)[[numbers]]
            initial_uset = uset[0]
            print(f'Initial loss = {v_lossf(uset)}')
        
    with tqdm(desc='Optimizing voltages') as pbar:
        with tqdm(desc='Optimizing voltages', disable=True) as pbarn:
            res = minimize(v_lossf, initial_uset, method='L-BFGS-B', 
                           callback=callback, tol = tol, bounds = voltage_bounds, options = {'maxiter' : 1000})
            loss = res.fun
            if loss > eps:
                warnings.warn(f"\nFrequency mismatch above accuracy treshold {eps}\nTry decreasing tol or apply find_initial_guess=True or increase call_num\nOptimization result: {res}")
            
            print(f'Final loss = {res.fun}')
            uset = res.x
            start_dcset[[numbers]] = uset
            dcset = start_dcset
            print(f'Final uset = {list(dcset)}')

    return dcset

def mode_optimization(s, alpha, potential_minimum, start_dcset, numbers=0, eps=1e-9, tol=1e-18, 
                      callback_num=1000, micro=1, voltage_bounds=(-15,15)):
    """
    Optimizes set of voltages on specified electrodes, so that the radial modes are rotated by the specified angle. 
    Conserves the potential minimum in the initial position. 

    Parameters
    ----------
    s : electrode.System object
        System object from *electrode* package, defining the trap
    alpha : float
        Angle of mode rotation in rad.
    potential_minimum : array shape([3])
        Coordinates of RF potential minimum.
    start_dcset : array shape([number of dc electrodes])
        Starting DC voltages for the optimization (on all electrodes).
    numbers : list shape([number of optimizing electrodes]) or int, optional, default is 0
        list of the indeces of DC electrodes, used for optimization. If 0, all DC elecrtodes are used for optimization.
    eps : float, optional, default is 1e-9
        Maximal loss, at which the algorithm considers the implementation succesful. If lower eps are possible, 
        the algorithm will keep optimizing according to tol.
    tol : float, optional, default is 1e-18
        Tolerance for termination from scipy.optimize.
    callback_num : int, optional, default is 1000
        Number of iterations, at which the current voltage set will be as callback printed.
    micro : float, optional, default is 1
        If non 0, algorithm will attempt to compensate micromotion by keeping potential minima at the starting positions.
        Coeff value coresponds to the input of minimum displacement in loss function, and should be evaluated emperically.
        Approximate range: 0.001 to 10
    voltage_bounds : tuple (float,float) or list of tuples len(number of dc electrodes), optional, default is (-15,15)
        List of voltage bounds on each DC electrode. If tuple, then each electrode will be assigned the same bounds. 
        Alternatively, each electrode can be assigned individual bounds as [(-bnds_1, bnds_1), (-bnds_2,bnds_2),...]

    Returns
    -------
    dcset : list shape([number of dc electrodes])
        Resulting DC voltage set.

    """
    try:
        numbers[0]
    except:
        numbers = np.arange(len(start_dcset))
        
    numbers = np.array(numbers)
    start_dcset = np.array(start_dcset)
    
    def v_lossf(uset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start_dcset[[numbers]] = uset
            pbarn.update(1)
            loss = 0
            u_set = np.zeros(len([rfs for rfs in s.rfs if rfs != 0]))
            u_set = np.concatenate([u_set, start_dcset])
            
            with s.with_voltages(dcs=u_set, rfs=None):
                try:
                    x1 = s.minimum(potential_minimum*1.001, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG",
                                   tol = 1e-6, options = {'maxiter' : 2000})
                    _, mod_dir = s.modes(x1, sorted=False)
                    
                    e1 = np.array([0,0,1])
                    alp = np.arccos(np.inner(mod_dir[2],e1))
                    
                    loss += (alpha - alp)**2
                    loss += micro * np.linalg.norm(x1 - potential_minimum)**2
    
                except: 
                    loss += 10

        return loss

    count=[0]
    def callback(intermediate_result):
        pbar.update(1)
        count[0] += 1
        if count[0] % callback_num == 0:
            print(f'Current loss = {intermediate_result.fun}')
            print(f'Current uset = {list(intermediate_result.x)}')
        return count

    try:
        voltage_bounds[0][0]
        voltage_bounds=list(voltage_bounds)
    except:
        voltage_bounds = [(voltage_bounds[0],voltage_bounds[1]) for n in range(len(numbers))]

    with tqdm(desc='Optimizing voltages', disable=True) as pbarn:
        uset = np.array(start_dcset)[[numbers]]
        initial_uset = uset[0]
        print(f'Initial loss = {v_lossf(uset)}')
    
    with tqdm(desc='Optimizing voltages') as pbar:
        with tqdm(desc='Optimizing voltages', disable=True) as pbarn:
            res = minimize(v_lossf, initial_uset, method='L-BFGS-B', callback=callback, tol = tol, 
                           bounds = voltage_bounds, options = {'maxiter' : 10000})
            loss = res.fun
            if loss > eps:
                warnings.warn(f"\nMode mismatch above accuracy treshold {eps}\nTry decreasing tol or change micro or change initial guess or change trap geometry\nOptimization result: {res}")
            
            print(f'Final loss = {res.fun}')
            uset = res.x
            start_dcset[[numbers]] = uset
            dcset = start_dcset
            print(f'Final uset = {list(dcset)}')
    
    return dcset

def position_optimization(s, positions, start_dcset, numbers=0, eps=1e-2, tol=1e-18,
                          ion_masses=1, charges=1, callback_num=100, voltage_bounds=(-15,15)):
    """
    Optimizes set of voltages on specified electrodes, so that the potential minimum positions match the desired ones.

    Parameters
    ----------
    s : electrode.System object
        System object from *electrode* package, defining the trap
    positions : list shape([number of positions, 3])
        Positions of potential minima, at which ion frequencies are calculated
    start_dcset : list shape([number of dc electrodes])
        Starting DC voltages for the optimization (on all electrodes).
    numbers : list shape([number of optimizing electrodes]) or int, optional, default is 0
        list of the indeces of DC electrodes, used for optimization. If 0, all DC elecrtodes are used for optimization.
    eps : float, optional, default is 1e-2
        Maximal loss, at which the algorithm considers the implementation succesful. If lower eps are possible, 
        the algorithm will keep optimizing according to tol.
    tol : float, optional, default is 1e-18
        Tolerance for termination from scipy.optimize.
    ion_masses : np.array shape([ion number]) or float, optional, default is 1
        Either array of ion masses for mixed species crystal, or a single ion mass for a single species crystal
    charges : array shape([ion_number]) or int, optional, default is 1
        Array of ion charges for crystals with varying charged, or a single charge for uniformly charged ions. 
        Scaled for elementary charge. 
    callback_num : int, optional, default is 100
        Number of iterations, at which the current voltage set will be as callback printed.
    voltage_bounds : tuple (float,float) or list of tuples len(number of dc electrodes), optional, default is (-15,15)
        List of voltage bounds on each DC electrode. If tuple, then each electrode will be assigned the same bounds. 
        Alternatively, each electrode can be assigned individual bounds as [(-bnds_1, bnds_1), (-bnds_2,bnds_2),...]

    Returns
    -------
    dcset : list shape([number of dc electrodes])
        Resulting DC voltage set.

    """
    positions = np.array(positions)
    try:
        numbers[0]
    except:
        numbers = np.arange(len(start_dcset))
    
    try:
        M = ion_masses[0]
    except:
        ion_masses = np.ones(positions.shape[0])*ion_masses
        M = ion_masses[0]
    try:
        Z = charges[0]
    except:
        charges = np.ones(positions.shape[0]) * charges
        Z = charges[0]
    
    rf_set = s.rfs
    numbers = np.array(numbers)
    start_dcset = np.array(start_dcset)

    def v_lossf(uset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start_dcset[[numbers]] = uset
            pbarn.update(1)
            loss = 0
            u_set = np.zeros(len([rfs for rfs in s.rfs if rfs != 0]))
            u_set = np.concatenate([u_set, start_dcset])
            
            for i, pos in enumerate(positions):
                with s.with_voltages(dcs=u_set, rfs=rf_set*np.sqrt(M*charges[i]/ion_masses[i]/Z)):
                    try:
                        x1 = s.minimum(pos*1.001, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG",
                                    tol = 1e-9, options = {'maxiter' : 2000})                        
                        loss += np.linalg.norm(x1-pos)**2
        
                    except: 
                        loss += 1e6

        return loss

    count=[0]
    def callback(intermediate_result):
        pbar.update(1)
        count[0] += 1
        if count[0] % callback_num == 0:
            print(f'Current loss = {intermediate_result.fun}')
            print(f'Current uset = {list(intermediate_result.x)}')
        return count

    try:
        voltage_bounds[0][0]
        voltage_bounds=list(voltage_bounds)
    except:
        voltage_bounds = [(voltage_bounds[0],voltage_bounds[1]) for n in range(len(numbers))]

  
    with tqdm(desc='Optimizing voltages', disable=True) as pbarn:
        uset = np.array(start_dcset)[[numbers]]
        initial_uset = uset[0]
        print(f'Initial loss = {v_lossf(uset)}')
    
    with tqdm(desc='Optimizing voltages') as pbar:
        with tqdm(desc='Optimizing voltages', disable=True) as pbarn:
            res = minimize(v_lossf, initial_uset, method='L-BFGS-B', 
                           callback=callback, tol = tol, bounds = voltage_bounds, options = {'maxiter' : 10000})
            loss = res.fun
            if loss > eps:
                warnings.warn(f"\nPosition mismatch above accuracy treshold {eps}\nTry decreasing tol or change start_dcset or change geometry\nOptimization result: {res}")
            
            print(f'Final loss = {res.fun}')
            uset = res.x
            start_dcset[[numbers]] = uset
            dcset = start_dcset
            print(f'Final uset = {list(dcset)}')
    
    return dcset
