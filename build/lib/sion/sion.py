from pylion.lammps import lammps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from electrode import (System, PolygonPixelElectrode, PointPixelElectrode)
import sys
import scipy

ech = 1.602176634e-19  # electron charge, C
amu = 1.66053906660e-27  # atomic mass unit, kg
eps0 = 8.8541878128e-12  # vacuum electric permittivity

"""
Functions, simulating the ion dynamics above planar traps
"""

@lammps.fix
def polygon_trap(uid, Omega, rf_voltages, dc_voltages, RFs, DCs, cover=(0, 0)):
    """
    Simulates an arbitrary planar trap with polygonal electrodes

    :param uid: str
        ions ID
    :param Omega: list shape len(RFs)
        array of RF frequencies of each RF electrode
    :param u_set: array shape (len(RFs) + len(DCs))
        Set of the (peak) voltages on RF and DC electrodes.
        Peak RF voltages come first, followed by the DC voltages.
    :param RFs: list shape (len(RFs), 4)
        Array of coordinates of RF electrodes in m
    :param DCs: list shape (len(DCs), 4)
        Array of coordinates of DC electrodes in m
        The order of electrodes must match the order of u_set and Omegas.
    :param cover: list shape (2)
        array [cover_number, cover_height] - number of the terms in
        cover electrode influence expansion and its height.

    :return: str
        updates simulation lines, which will be executed by LAMMPS
    """
    odict = {}
    Omega = np.array(Omega)
    lines = [
        f'\n# Creating a polygonal Surface Electrode Trap... (fixID={uid})']
    odict['timestep'] = 1 / np.max(Omega) / 20
    rf = [len(a) for a in RFs]
    nrf = len(rf)
    dc = [len(a) for a in DCs]
    ndc = len(dc)
    for i in range(nrf):
        lines.append(
            f'variable phase{uid}{i:d}\t\tequal "{Omega[i]:e}*step*dt"')
    xcc = []
    ycc = []
    zcc = []
    u_set = np.concatenate((np.array(rf_voltages), np.array(dc_voltages)))
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
                z = f'z'
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
                        nothing = 0
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
                x1 = x2
                y1 = y2
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
                        nothing = 0
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

    :param: Omega: list shape (number of RF points)
        array of RF frequencies of all RF points.
    :param: trap: list shape (4, shape(element))
        trap = [RF: elec: [omega, v, areas, coords], DC: elec: [v, areas, coords]]
        list of the shape [RF_areas, RF_coordinates, DC_areas, DC_coordinates],
        where areas represent the areas of each point, and coordinates represent the
        coordinates of each point center.
    :param: voltages: list (2, number of electrodes)
        list of arrays [RF_voltages, DC_voltages], where RF_voltages
        are the peak voltage of RF points, and DC_voltages - voltages of each DC point.
    :param: cover: [cover_max, cover_height].

    :return: str
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
    
    # check if areas of points are equal
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
                    z = f'z'
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
                    z = f'z'
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

    :param: Omega: float
        RF frequency of the ring
    :param: r: float
        inner radius of the ring
    :param: R: float
        outer radius of the ring
    :param: voltage: float
        RF amplitude of the ring
    :param: resolution: int
        number of points in the ring design
    :param: center: list shape (2)
        coordinate of the center of the ring
    :param: cover: [cover_max, cover_height].

    :return: str
        updates simulation of a single ring

    """
    s, omega, trap, volt = ring_trap_design(RF_voltage, Omega, r, R, r_dc, dc_voltage, res = resolution, need_coordinates = True, cheight=cover[0], cmax=cover[1])

    return point_trap(trap, cover)


"""
Shuttling of ions in the trap
"""

def q_tan(t, d, T, N):
    return d/2*(np.tanh(N*(2*t-T)/T)/np.tanh(N) + 1)

def lossf_shuttle(uset, s, omega0, x, L):
    """
    Loss function for search of the optimal voltage sequence

    """
    u_set = [0 for e in s if e.rf] 
    u_set.extend(uset)
    loss = 0
    attempts = 0
    with s.with_voltages(dcs = u_set, rfs = None):
        while attempts < 200:
            try:
                xreal = s.minimum(x + np.array([(-1)**attempts*attempts/10, 0, 0]), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
                break
            except:
                attempts += 1
                xreal = x

        curv_z, mod_dir=s.modes(xreal,sorted=False)
        loss += np.linalg.norm(xreal - x) + L**(-2)*(curv_z[0]-omega0)**2
        #+ ((omega - omega0))**2
    return loss

def linear_shuttling_voltage(s, x0, d, T, N, u_set, vmin = -15, vmax = 15, res = 50, L = 1e-6):
    """
    Performs optimization of voltage sequence on DC electrodes for the linear
    shuttling, according the tanh route. Voltage is optimized to maintain 
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
    N : int
        parameter in the tanh route definition (see q_tan)
    u_set : list shape (len(RFs)+len(DCs))
        list of starting DC voltages (including zeros for RF voltages)
    vmin : float, optional
        Minimal allowable DC voltage. The default is -15.
    vmax : float, optional
        Maximal allowable DC voltage. The default is 15.
    res : int, optional
        Resolution of the voltage sequence determination. The default is 50.
    L : float, optional
        Length scale of the electrode. The default is 1e-6 which means mkm

    Returns
    -------
    voltage_seq : list shape (len(DCs), res+1)
        voltage sequences on each DC electrode (could be constant)

    """
    voltage_seq = []
    bnds = [(vmin,vmax) for el in s if not el.rf]
    with s.with_voltages(dcs = u_set, rfs = None):
        x1 = s.minimum(np.array(x0), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
        curv_z, mod_dir=s.modes(x1,sorted=False)
    for dt in range(res+1):
        t = dt*T/res
        x = x0 + np.array([q_tan(t, d, T, N), 0, 0])
        uset = [el for el in u_set if not el == 0]
        uset = scipy.optimize.minimize(lossf_shuttle, uset, args = (s, curv_z[0], x, L), tol = 1e-6, bounds = bnds, options = {'maxiter' : 100000})
        voltage_seq.append(uset.x)
        u_set = uset.x
    voltage_seq = np.array(voltage_seq).T
    return voltage_seq

def fitter_tan(t, a, b, c, d):
    return a*np.tanh(b*t + c) + d

def fitter_norm(t, a, b, c, d):
    return a*np.exp(b*(t-c)*2) + d

def approx_shuttling(voltage_seq, T, res = 50):
    """
    Approximation of voltage sequences on DC electrodes with analytic functions

    Parameters
    ----------
    voltage_seq : list shape (len(DCs), res+1)
        voltage sequences on each DC electrode (could be constant)
    T : float
        Time of shuttling operation
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
    for seq in voltage_seq:
        mean = np.mean(seq)
        dif = seq[-1] - seq[0]
        lin = np.std(seq)/mean
        att = 0
        if lin < 1e-4:
            funcs.append('(%5.3f)' % mean)
        else:
            try:
                popt_tan, pcov_tan = scipy.optimize.curve_fit(fitter_tan, x_data, seq, [np.abs(dif/2), dif/T, -dif/2, mean])
                tan = np.linalg.norm(seq - fitter_tan(x_data, *popt_tan))
            except:
                att += 1
                tan = 1000
            try:
                popt_norm, pcov_norm = scipy.optimize.curve_fit(fitter_norm, x_data, seq)
                norm = np.linalg.norm(seq - fitter_norm(x_data, *popt_norm))
            except:
                att +=1
                norm = 1000
            if att == 2:
                sys.exit("Needs custom curve fitting")
            
            if tan > norm:
                funcs.append('((%5.3f) * exp((%5.3f) * (step*dt - (%5.3f))^2) + (%5.3f))' % tuple(popt_norm))
            else:
                funcs.append('((%5.3f) * (1 - 2/(exp(2*((%5.3f) * step*dt + (%5.3f))) + 1)) + (%5.3f))' % tuple(popt_tan))

    return funcs

@lammps.fix
def polygon_shuttling(uid, Omega, rf_set, RFs, DCs, shuttlers, cover=(0, 0)):
    """
    Simulates an arbitrary planar trap with polygonal electrodes
    and shuttling in this trap. The shuttling is specified by the 
    voltage sequence on the DC electrodes. 

    :param uid: str
        ions ID
    :param Omega: list shape len(RFs)
        array of RF frequencies of each RF electrode
    :param rf_set: array shape (len(RFs))
        Set of the (peak) voltages on RF electrodes.
    :param RFs: list shape (len(RFs), 4)
        Array of coordinates of RF electrodes in m
    :param DCs: list shape (len(DCs), 4)
        Array of coordinates of DC electrodes in m
    :param shuttlers: list of strings shape (len(shuttlers))
        list of strings, representing functions at which voltage on DC electrodes
        is applied through simulation time
    :param cover: list shape (2)
        array [cover_number, cover_height] - number of the terms in
        cover electrode influence expansion and its height.

    :return: str
        updates simulation lines, which will be executed by LAMMPS
    """
    odict = {}
    Omega = np.array(Omega)
    lines = [
        f'\n# Creating a polygonal Surface Electrode Trap... (fixID={uid})']
    odict['timestep'] = 1 / np.max(Omega) / 20
    rf = [len(a) for a in RFs]
    nrf = len(rf)
    dc = [len(a) for a in DCs]
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
                    nothing = 0
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
                    nothing = 0
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
    x = np.vstack([[i + j*.5, j*3**.5*.5] for j in range(-res - min(0, i), res - max(0, i) + 1)] for i in range(-res, res + 1))/(res + .5)*scale # centers
    
    a = np.ones(len(x))*3**.5/(res + .5)**2/2*scale**2 # areas
    
    ps = []
    ars = []
    
    for i, el in enumerate(x):
        if boundary(n, el):
            ps.append(el)
            ars.append(a[i])
    return np.array(ps), np.array(ars)


def five_wire_trap_design(Urf, DCtop, DCbottom, cwidth, clength, boardwidth, rftop, rflength, rfbottom, need_coordinates = False, patternTop = 1, patternBot = 1, getCoordinate = False, gapped = 0, cheight=0, cmax=0, need_plot = False):
    """
    A function for a very detailed definition of five wire traps.It allows to 
    define side DC electrodes only by their width and height, and the number
    of repeating electrodes of the same shapes. It also allows to obtain the 
    coordinates with the respect to the gap width between electrodes.
    If specified, a plot of the trap layout may be deomnstrated. 

    :param Urf: float
        A parametrized RF voltage for pseudopotential approximation 
        Urf = Vrf*np.sqrt(Z/M)/(2*L*Omega)
        For Vrf - peak RF voltage in V, Z - ion charge, M - ion mass 
    :param DCtop: list shape (number of upper DC electrodes, 2)
        An array of [[height, width],...] of a repetetive cell
        of upper DC electrodes
    :param DCbottom: list shape (number of lower DC electrodes, 2)
        An array of [[height, width], []] of a repetetive cell
        of lower DC electrodes
    :param cwidth: float
        A width of central DC electrode
    :param clength: float
        A length of central DC electrode, placed symmetrical to y-axis
    :param boardwidth: float
        A gap width between each electrodes
    :param rftop: float
        Width of upper RF electrode
    :param rflength: float
        Length of RF line, placed symmetrically to y-axis
    :param rfbottom: float
        Width of lower RF electrode
    :param patternTop: int
        Number of repetitions of upper DC cell, if None then ==1
    :param patternBot: int
        Number of repetitions of lower DC cell
    :param getCoordinate: int
        If not None, the coordinates of electrodes with gaps will
        be written to a file coordinates.txt
    :param gapped: float
        An experimental parameter of gaps between RF electrodes and DC
        If None then ==0
    :param cheight: float
        A height of a cover electrode plate, which will be considered
        if these params are not 0
    :param cmax: int
        A number of cover potential expansion, it's often accurate
        to assume cmax = 5. If 0, then it is not considered
    :param plott: int
        if not None, then creates a plot of this trap
    
    :return: electrodes: list shape (number of electrodes, 2, shape of electrode)
        electrode coordinates for sion simulation with five_wire_trap()
    :return: N: int
        number of all DC electrodes
    :return: s: electrode.System object
        system object for electrode package.
    """
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

    # Reverse every electrode, because apparently author intended them to be defined conterclockwise, but never bothered to state this fact
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
        ("", [np.array(rf_top) * 1e-6,
                np.array(rf_bottom) * 1e-6])]
    for i in range(n):
        st = "t[" + str(i + 1) + "]"
        elec.append([st, [np.array(t[i]) * 1e-6]])
    for i in range(nb):
        st = "b[" + str(i + 1) + "]"
        elec.append([st, [np.array(b[i]) * 1e-6]])
    elec.append(["c", [np.array(c) * 1e-6]])

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
        # u = s.rfs sets the voltage-type for the voltage plot to RF-voltages (DC are not shown)
        xmax = rflength * 2 / 3
        ymaxp = (np.max(DCtop) + rftop + clefttop[1]) * 1.2
        ymaxn = (np.max(DCbottom) + rfbottom) * 1.2
        ax[0].set_title("colour")
        # ax[0] addresses the first plot in the subplots - set_title gives this plot a title
        ax[1].set_title("rf-voltages")
        for axi in ax.flat:
            axi.set_aspect("equal")
            axi.set_xlim(-xmax, xmax)
            axi.set_ylim(-ymaxn, ymaxp)
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


def ring_trap_design(Urf, Omega, r, R, r_dc = 0, v_dc = 0, res = 100, need_coordinates = False, cheight=0, cmax=0):
    """
    Returns an array of RF rings placed in a row, with a DC circle in each center.

    :param r: float
        inner radius of each ring
    :param R: float
        outer radius of each ring
    :param res: int
        resolution - a number of points defining the ring
    :param dif: float
        a distance between ring centers
    :param Nring: int
        a number of rings
    :param rdc: float
        if not None - a radius of central DC electrode in each ring.

    :return: x: list shape (number of points, 2)
        array of coordinates of center of each point
    :return: a: list shape (number of points)
        array of areas of each point
    :return: count: list 
        tuple of the form [count_rf, count_dc], which gives number
        of all RF and DC points, respectively. 
    """
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
    
    return point_trap_design(frequencies, rf_voltages, dc_voltages, ring_boundary, scale, res, need_coordinates, cheight=cheight, cmax=cmax)

def point_trap_design(frequencies, rf_voltages, dc_voltages, boundaries, scale, resolution, need_coordinates = False, need_plot = False, cheight=0, cmax=0):
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
        #plot of the ring trap, which will demonstrate the accuracy of chosen resolution
        fig, ax = plt.subplots(1,2,figsize=(13, 5))
        s.plot_voltages(ax[0], u=s.rfs)
        ax[0].set_xlim((-scale, scale))
        ax[0].set_ylim((-scale, scale))
        ax[0].set_title("RF voltage")
        s.plot_voltages(ax[1], u=s.dcs)
        ax[1].set_title('dc voltage')
        ax[1].set_xlim((-scale, scale))
        ax[1].set_ylim((-scale, scale))
        cmap = plt.cm.RdBu_r
        norm = mpl.colors.Normalize(vmin=np.min(dc_voltages), vmax=np.max(dc_voltages))

        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, shrink =0.9)

        """cb = plt.colorbar(ax=ax[2], shrink=0.9)"""
        cb.ax.tick_params(labelsize=8)
        cb.set_label('Voltage', fontsize = 8)

        plt.show()

    if need_coordinates:
        return s, trap
    else:
        return s


def n_rf_trap_design(Urf, DCtop, DCbottom, cwidth, rfwidth, rflength, n_rf=1, L = 1e-6, patternTop=1,
              patternBot=1, cheight=0, cmax=0, need_coordinates = False, need_plot = False):
    """
    A function for convenient definition of multi-wire traps, with n asymmetrical RF lines.

    :param Urf: list shape (n_rf, 2)
        A parametrized RF voltage set for pseudopotential approximation
        The voltages are given for each pair of RF electrodes as [Urf_upper, Urf_lower]
        Urf = [Urf_inner_upper, Urf_inner_lower, Urf_outer_upper, Urf_outer_lower]
    :param DCtop: list shape (number of upper DC electrodes, 2)
        An array of [[height, width],...] of a repetetive cell
        of upper DC electrodes
    :param DCbottom: list shape (number of lower DC electrodes, 2)
        An array of [[height, width], []] of a repetetive cell
        of lower DC electrodes
    :param cwidth: float
        A width of central DC electrode. Its length is equal to rflength
    :param rflength: float
        Length of RF line, placed symmetrically to y-axis
    :param rfwidth: list shape (n_rf, 2)
        list of widthes of an RF electrode (top, bottom), forming the RF lines
    :param patternTop: int
        Number of repetitions of upper DC cell, if None then ==1
    :param patternBot: int
        Number of repetitions of lower DC cell
    :param cheight: float
        A height of a cover electrode plate, which will be considered
        if these params are not 0
    :param cmax: int
        A number of cover potential expansion, it's often accurate
        to assume cmax = 5. If 0, then it is not considered
    :param plott: int
        if not None, then creates a plot of this trap
    
    :return: electrodes: list shape (2, shape of RF/DC electrodes)
        electrode coordinates for sion simulation with polygon_trap()
    :return: s: electrode.System object
        system object for electrode package.
    """

    c = [[-rflength/2, 0], [rflength/2, 0],
         [rflength/2, cwidth], [-rflength/2, cwidth]]
    RF = []
    electrodes = []
    for i in range(n_rf):
        rf_top = [[-rflength/2, cwidth+i*rfwidth[i][0]], [rflength/2, cwidth+i*rfwidth[i][0]],
                  [rflength/2, cwidth+(i+1)*rfwidth[i][0]], [-rflength/2, cwidth+(i+1)*rfwidth[i][0]]]
        rf_bottom = [[-rflength / 2,  -(i+1) * rfwidth[i][1]], [rflength / 2, -(i+1) * rfwidth[i][1]],
                     [rflength / 2, -i * rfwidth[i][1]], [-rflength / 2, -i * rfwidth[i][1]]]
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
    # starting point - central electrode, among top DCs. Considering parity of electrode number (i just want it to be BEAUTIFUL)
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

    # Reverse every electrode, because apparently author intended them to be defined conterclockwise, but never bothered to state this fact
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

    # Polygon approach. All DCs are 0 for nown
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
    # Exact coordinates for lion

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
            
    if need_coordinates:
        return s, RF, DC
    else:
        return s

def polygons_from_gds(gds_lib, L = 1e-6, need_plot = False, need_coordinates = False, cheight=0, cmax=0):
    try:
        import gdspy
    except:
        sys.exit("gdspy is not installed.")
    lib = gdspy.GdsLibrary(infile=gds_lib)
    count = 0
    full_elec = []
    electrodes = []

    for cell in lib.top_level():
        for coordinates in cell.get_polygons():
            electrodes.append([f'[{count}]', [coordinates]])
            count+=1
            full_elec.append(np.array(coordinates)*L)
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
            
    if need_coordinates:
        return s, full_elec
    else:
        return s

    

def individual_wells(Urf, width, length, dot, x, y, L = 1e-6):
    """
    
    Polygonal RF electrode with arbitrary rectangular notch generation in the
    determined positions for individual potential wells and geometry optimization.
    
    :param: Urf: float
        Parametrized pseudopotential voltage
    :param: width: float
        width of the RF electrode (vertical)
    :param: length: float
        length of the RF electrode (horizontal)
    :param: dot: list shape (number of dots, 2)
        on-chip coordinates of the centers of individual wells
    :param: x: list shape (number of dots)
        array of (horizontal) widthes of the notches
    :param: y: list shape (number of dots)
        array of (vertical) legthes of the notches
    :param: L: float, optional
        Length scale in definition of the trap. The default is 1e-6. This means, 
        trap is defined in mkm. 
    
    :return: s: electrode.System object
        System object for electrode package simulation
    :return: electrodes: list shape (2, shape of electrodes)
        coordinates of RF electrodes for Sion simulation
    """
    n = dot.shape[0]
    
    rfstart = [[-length/2, -width/2], [dot[0,0]-x[0]/2, -width/2], [dot[0,0]-x[0]/2, width/2], [-length/2, width/2]]
    rfs = [rfstart]
    
    for i in range(n-1):
        rfs.append([[dot[i, 0]-x[i]/2, dot[i, 1]+y[i]/2], [dot[i, 0]+x[i]/2, dot[i, 1]+y[i]/2], [dot[i, 0]+x[i]/2, width/2], [dot[i, 0]-x[i]/2, width/2]])
        rfs.append([[dot[i, 0]-x[i]/2, -width/2],[dot[i, 0]+x[i]/2, -width/2], [dot[i, 0]+x[i]/2, dot[i, 1]-y[i]/2], [dot[i, 0]-x[i]/2, dot[i, 1]-y[i]/2]])
        rfs.append([[dot[i,0]+x[i]/2, -width/2], [dot[i+1,0]-x[i+1]/2, -width/2], [dot[i+1,0]-x[i+1]/2, width/2],[dot[i,0]+x[i]/2, width/2]])
    
    i = n-1
      
    rfs.append([[dot[i, 0]-x[i]/2, dot[i, 1]+y[i]/2], [dot[i, 0]+x[i]/2, dot[i, 1]+y[i]/2], [dot[i, 0]+x[i]/2, width/2], [dot[i, 0]-x[i]/2, width/2]])
    rfs.append([[dot[i, 0]-x[i]/2, -width/2],[dot[i, 0]+x[i]/2, -width/2], [dot[i, 0]+x[i]/2, dot[i, 1]-y[i]/2], [dot[i, 0]-x[i]/2, dot[i, 1]-y[i]/2]])
    rfs.append([[dot[i,0]+x[i]/2, -width/2], [length/2, -width/2], [length/2, width/2],[dot[i,0]+x[i]/2, width/2]])

    electrodes = [
        ("rf", np.array(rfs))]

    
    s = System([PolygonPixelElectrode(name=f, paths=map(np.array, p))
                for f, p in electrodes])
    
    s[0].rf = Urf
    
    return s, [np.array(rfs)*L, [[[0,0],[1e-6,1e-6], [2e-6, 2e-6], [3e-6,3e-6]]]]




"""
Useful tools for initializing the simulation
"""

def ioncloud_min(x, number, radius):
    """
    Generates positions ions in of random ion cloud near the given point

    :param x: list shape (3)
        coordinates of a given point
    :param number: int
        a number of ions in the cloud
    :param radius: float
        a radius of cloud
    
    :return: list shape (number, 3)
        list of positions of ions 
    """
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
    """
    Positions of ions placed in linear order near the given point

    :param x: list shape (3)
        coordinates of given point
    :param number: int
        a number of ions in the cloud
    :param dist: float
        distance between placed ions
    
    :return: list shape (number, 3)
        list of positions of ions 
    """
    positions = []
    xini = x - np.array([dist * number / 2, 0, 0])
    for ind in range(number):
        xi = xini + np.array([dist * ind, 0, 0])
        positions.append(xi)

    return positions

"""
Normal mode calculation
"""

def axial_linear_hessian_matrix(ion_positions, omega_sec, Z, mass):
    """
    Calculates axial hessian and M_matrix (x-ax) of a linear ion chain

    :param: ion_positions: np.ndarray [ion_number, 3]
        array of final ion positions
    :param: omega_sec: float
        axial secular frequency of the trap in potential minimum
    :param: Z: float
        ion charge in C
    :param: mass: ion mass in kg
    
    :return: list (A_matrix, M_matrix): 
        A_matrix is axial hessian matrix, M - matrix of ion masses

    """
    global amu, ech, eps0
    ion_number = ion_positions.shape[0]
    axial_freqs = np.array([omega_sec for x in range(ion_number)])
    ions_mass = np.array([mass for x in range(ion_number)])
    ions_charge = np.array([Z for x in range(ion_number)])
    interaction_coeff = 1 / 4 / np.pi / eps0
    A_matrix = np.diag(ions_mass * axial_freqs ** 2 * 4 * np.pi ** 2)
    for i in range(ion_number):
        S = 0
        for j in range(ion_number):
            if j != i:
                A_matrix[i, j] = -2 * ions_charge[i] * ions_charge[j] * interaction_coeff / np.abs(
                    ion_positions[i, 0] - ion_positions[j, 0]) ** 3
                S += 2 * ions_charge[i] * ions_charge[j] * interaction_coeff / np.abs(
                    ion_positions[i, 0] - ion_positions[j, 0]) ** 3
        A_matrix[i, i] += S
    M_matrix = np.diag(ions_mass ** (-0.5))
    return (A_matrix, M_matrix)


def radial_linear_hessian_matrix(ion_positions, omega_sec, Z, mass):
    """
    Calculates radial (y-ax) hessian and M-matrix of a linear ion chain

    :param: ion_positions: np.ndarray [ion_number, 3]
        array of final ion positions
    :param: omega_sec: float
        secular frequency of the trap in minimum
    :param: Z: float
        ion charge in C
    :param: mass: float
        ion mass in kg
    
    :return: list (B_matrix, M_matrix)
        B_matrix is hessian matrix.
        """
    global amu, ech, eps0
    ion_number = ion_positions.shape[0]
    radial_freqs = np.array(np.array([omega_sec for x in range(ion_number)]))
    ions_mass = np.array([mass for x in range(ion_number)])
    ions_charge = np.array([Z for x in range(ion_number)])
    interaction_coeff = 1 / 4 / np.pi / eps0
    B_matrix = np.diag(ions_mass * radial_freqs ** 2 * 4 * np.pi ** 2)
    for i in range(ion_number):
        S = 0
        for j in range(ion_number):
            if j != i:
                B_matrix[i, j] = ions_charge[i] * ions_charge[j] * interaction_coeff / np.abs(
                    ion_positions[i, 0] - ion_positions[j, 0]) ** 3
                S -= ions_charge[i] * ions_charge[j] * interaction_coeff / np.abs(
                    ion_positions[i, 0] - ion_positions[j, 0]) ** 3
        B_matrix[i, i] += S
    M_matrix = np.diag(ions_mass ** (-0.5))
    return (B_matrix, M_matrix)


def linear_axial_modes(ion_positions, omega_sec, Z, mass):
    """
    Calculates axial normal modes of a linear ion chain

    :param: ion_positions: np.ndarray [ion_number, 3]
        array of final ion positions
    :param: omega_sec: float
        axial secular frequency of the trap in minimum
    :param: Z: float
        ion charge in C
    :param: mass: float
        ion mass in kg
    
    :return: (freq: list shape (ion number), normal_vectors: list shape (ion number, ion number)):
        (freq, normal_vectors) arrays of normal frequencies and respective modes
    """

    global amu, ech, eps0
    axial_hessian, mass_matrix = axial_linear_hessian_matrix(
        ion_positions, omega_sec, Z, mass)
    D = mass_matrix.dot(axial_hessian.dot(mass_matrix))
    freq, normal_vectors = np.linalg.eigh(D)
    normal_vectors = -normal_vectors.T
    freq = freq ** 0.5 / 2 / np.pi
    return (freq, normal_vectors)


def linear_radial_modes(ion_positions, omega_sec, Z, mass):
    """
    Calculates radial normal modes of a linear ion chain, for y' or z' --
    rotated radial secular modes, depending on the omega_sec parameter

    :param: ion_positions: np.ndarray [ion_number, 3]
        array of final ion positions
    :param: omega_sec: float
        axial secular frequency of the trap in minimum
    :param: Z: float
        ion charge in C
    :param: mass: float
        ion mass in kg
    
    :return: (freq: list shape (ion number), normal_vectors: list shape (ion number, ion number)):
        (freq, normal_vectors) arrays of normal frequencies and respective modes
    """
    global amu, ech, eps0
    radial_hessian, mass_matrix = radial_linear_hessian_matrix(
        ion_positions, omega_sec, Z, mass)
    D = mass_matrix.dot(radial_hessian.dot(mass_matrix))
    freq, normal_vectors = np.linalg.eigh(D)
    normal_vectors = -normal_vectors.T
    freq = freq ** 0.5 / 2 / np.pi
    return (freq, normal_vectors)


def hessian(ion_positions, omega_sec, ion_masses):
    """
    Hessian of an arbitrary coulomb crystal configuration. It may include
    non-linear ion chains (2D, 3D crystals), mixed-species ions, ions, experiencing
    different secular frequencies (tweezers, arrays of microtraps).

    :param ion_positions: list shape (ion number, 3)
        final positions of ions in a crystal
    :param omega_sec: list shape (ion number)
        secular frequencies of ions 
        which must be in the same order as positions
    :param ion_masses: list shape (ion_number)
        ion masses in the same order
    
    :return: A_matrix: list shape (3*ion_number, 3*ion_number)
        hessian
    :return: M_matrix: list shape (3*ion_number, 3*ion_number)
        mass matrix
    """
    global amu, ech, eps0
    ion_positions = np.array(ion_positions)
    omega_sec = np.array(omega_sec)*2*np.pi
    l = (ech**2 / (4*np.pi*eps0*ion_masses[0]*(omega_sec[0][2])**2))**(1/3)
    ion_positions = ion_positions/l
    N = len([len(a) for a in ion_positions])
    M_matrix = np.diag(list(np.array(ion_masses)**(-0.5))*3)

    def d(i, j):
        return np.linalg.norm(ion_positions[i] - ion_positions[j])
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


def normal_modes(ion_positions, omega_sec, ion_masses):
    """
    Normal modes of an ion crystal with arbitrary configuration

    :param ion_positions: list shape (ion number, 3)
        final positions of ions in a crystal
    :param omega_sec: list shape (ion number)
        secular frequencies of ions 
        which must be in the same order as positions
    :param ion_masses: list shape (ion_number)
        ion masses in the same order
    
    :return: freqs: list shape (3*ion_number)
        1D array of normal frequencies, 
    :return: modes: list shape (3*ion_number, 3*ion_number)
        2D array of normal modes
    """
    A_matrix, M_matrix = hessian(ion_positions, omega_sec, ion_masses)
    AA = np.dot(M_matrix, np.dot(A_matrix, M_matrix))

    eig, norm_modes = np.linalg.eigh(AA)
    norm_modes = -norm_modes.T
    return np.sqrt(eig)/(2*np.pi), norm_modes


"""
Anharmonic parameters and modes
"""


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
        Length scale in definition of the trap. The default is 1e-6. This means, 
        trap is defined in mkm. 

    Returns
    -------
    results : list shape [number of dots, 3]
        array of anharmonic scale lengths (in m) in each calculated potential minimum. 
        Given as  [l2, l3, l4], for ln being scale length of the n's potential term

    """
    
    results = []
    for pos in minimums:
        x1 = s.minimum(np.array(pos), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
        curv_z, mod_dir=s.modes(x1,sorted=False)
        curv_z *= L**(-2)
        inv = np.linalg.inv(mod_dir)
        pot = s.potential(x1, derivative = 4)[0,axis,axis]
        eig_pot = np.dot(inv, np.dot(pot, mod_dir))
        eig = eig_pot[axis,axis]*L**(-4)
        k2 = curv_z[axis]*ech
        k4 = eig*ech
        pot3 = s.potential(x1, derivative = 3)[0,axis]
        eig_pot3 = np.dot(inv, np.dot(pot3, mod_dir))
        eig3 = eig_pot3[axis,axis]*L**(-3)
        k3 = eig3*ech
        l2 = (ech**2/(4*np.pi*eps0*k2))**(1/3)
        l3 = (k3/k2)**(1/(2-3))
        l4 = np.sign(k4)*(np.abs(k4)/k2)**(1/(2-4))
        results.append([l2, l3, l4])
    
    return results


def anharmonic_modes(s, ion_positions, masses, axis, poles = [1,1], L = 1e-6):
    """
    Anharmonic normal modes along the chosen pricniple axis of oscillation
    in a linear ion chain, considering hexapole and octopole anharmonicities.
    Defined for the mixed species ion crystals in individual potential wells

    Parameters
    ----------
    s : electrode.System object
        System object from electrode package, defining the trap
    ion_positions : list shape (ion_number, 3)
        list of final ion positions in crystal
    masses : list shape (ion_number)
        list of ion masses 
    axis : int
        axis, along which the normal modes are investigated. 
        Determined by respecting int: (0, 1, 2) = (x, y, z)
    poles : list shape (2)
        Determines which anharmonic terms to account. The default is 
        (1, 1), which means both hexapole and octopole anharmonicities are accounted
        if (1, 0) -- only hexapole (n = 3)
        if (0, 1) -- only octopole (n = 4)
    L : float, optional
        Length scale in definition of the trap. The default is 1e-6. 
        This means, trap is defined in mkm. 

    Returns
    -------
    freqs: list shape (ion_number)
        1D array of normal frequencies, 
    modes: list shape (ion_number, ion_number)
        2D array of normal modes

    """

    n = np.array(ion_positions).shape[0]
    alpha2 = []
    alpha3 = []
    alpha4 = []
    minims = []
    B = [[2, -2], [-1, 1], [-1, 1]]
    
    for pos in ion_positions:
        x1 = s.minimum(np.array(pos)/L, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
        minims.append(x1[0])
        curv_z, mod_dir=s.modes(x1,sorted=False)
        curv_z *= L**(-2)
        inv = np.linalg.inv(mod_dir)
        pot = s.potential(x1, derivative = 4)[0,axis,axis]
        eig_pot = np.dot(inv, np.dot(pot, mod_dir))
        eig = eig_pot[axis,axis]*L**(-4)
        k2 = curv_z[axis]*ech
        k4 = eig*ech
        pot3 = s.potential(x1, derivative = 3)[0,axis]
        eig_pot3 = np.dot(inv, np.dot(pot3, mod_dir))
        eig3 = eig_pot3[axis,axis]*L**(-3)
        k3 = eig3*ech
        l = (ech**2/(4*np.pi*eps0*k2))**(1/3)
        alpha2.append(k2)
        alpha3.append(l*k3/k2)
        alpha4.append(l**2*k4/k2)
    
    lo = (ech**2/(4*np.pi*eps0*alpha2[0]))**(1/3)
    ion_positions = np.array(ion_positions)/lo
    minims = np.array(minims)*L/lo
    hessian = np.zeros([n,n])
    for i in range(n):
        hessian[i,i] = alpha2[i]/alpha2[0] + poles[0]*2*alpha3[i]/(alpha2[i]/alpha2[0])*np.abs(ion_positions[i][0]-minims[i]) + poles[1]*6*alpha4[i]/(alpha2[i]/alpha2[0])*(ion_positions[i][0]-minims[i])**2
        coulomb = 0
        for p in range(n):
            if p != i:
                coulomb += B[axis][0]/(np.abs(ion_positions[i][0]-ion_positions[p][0]))**3
        hessian[i,i] += coulomb
    for i in range(n):
        for j in range(n):
            if i!=j:
                hessian[i,j] = B[axis][1]/(np.abs(ion_positions[i][0]-ion_positions[j][0]))**3
    for i, m in enumerate(masses):
        masses[i] = 1/m**0.5
    M_matrix = np.diag(masses)
    eig, vec = np.linalg.eig(np.dot(M_matrix, np.dot(hessian, M_matrix)))
    norm_modes = -vec.T
    return np.sqrt(eig*alpha2[0])/(2*np.pi), norm_modes


"""
Stability analysis
"""

def stability(s, Ms, Omega, Zs, minimum, L = 1e-6, need_plot = True):
    """
    Returns stability parameters for the linear planar trap (RF confinement only radial)
    If asked, return plot of the stability a-q diagram for this trap and plots 
    the parameters for this voltage configuration (as a red dot)

    Parameters
    ----------
    s : electrode.System object
        System object from electrode package, defining the trap
    M : float
        ion mass in kg
    Omega : float
        RF frequency 
    Z : float
        ion charge in C
    dot : list shape (3)
        approximate position of potential minimum, from which the
        real minimum is calculated
    L : float, optional
        Length scale in definition of the trap. The default is 1e-6. 
        This means, trap is defined in mkm. 
    plot : int, optional
        if plot == 1, then the plot of the stability diagram and the 
        a-q parameters for this voltage configuration is shown

    Returns
    -------
    params : list [areal, qreal, [alpha, thetha], [a_crit_lower, a_crit_upper], q_crit]
        Stability parameters for this trap, sorted as follows:
            areal: calculated a parameter for this trap and voltage set
            qreal: calculated q parameter for this trap and voltage set
            alpha: relation between y and z radial DC modes 
            thetha: angle of rotation between RF nad DC potential axes
            a_crit_upper: maximal achievable positive a parameter in this trap
            a_crit_lower: maximal achievable negative a parameter in this trap
            q_crit: maximal achievable q parameter in this trap
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

    qa_crit_upper = [(-cc - np.sqrt(cc**2 - 4*bb*(dd-aa)))/(2*(dd-aa)), (-cc + np.sqrt(cc**2 - 4*bb*(dd-aa)))/(2*(dd-aa))]
    qa_crit_lower = [(-ff - np.sqrt(ff**2 - 4*ee*(gg-hh)))/(2*(gg-hh)), (-ff + np.sqrt(ff**2 - 4*ee*(gg-hh)))/(2*(gg-hh))]
    if qa_crit_upper[0] < q_crit and qa_crit_upper[0] > 0:
        a_crit_upper = qa_crit_upper[0]**2/(2*alpha)
    if qa_crit_upper[1] < q_crit and qa_crit_upper[1] > 0:
        a_crit_upper = qa_crit_upper[1]**2/(2*alpha)
    if qa_crit_lower[0] < q_crit and qa_crit_lower[0] > 0:
        a_crit_lower = -qa_crit_lower[0]**2/2
    if qa_crit_lower[1] < q_crit and qa_crit_lower[1] > 0:
        a_crit_lower = -qa_crit_lower[1]**2/2

    params['Range of achievable a'] = [a_crit_lower, a_crit_upper]
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
    
        plt.show()
        
    return params

"""
Optimization functions

The optimization routine is written in Python, which affects its speed.
However, the main source of the slowness of these algorithms is a necessity 
to determine the potential minimum to calculate the loss function. 
The potential minimum is determined with its own optimization, which is already
written in C, so only minor increase in speed can be obtained by rewritting these functions in C. 
"""

def v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L = 1e-6):
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
    
    u_set = np.zeros(numbers[0])
    u_set = np.append(u_set, uset)
    
    #obtain result of the function
    with s.with_voltages(dcs = u_set, rfs = None):
        for i, pos in enumerate(dots):
            try:
                x1 = s.minimum(pos, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
                curv, mod_dir=s.modes(x1,sorted=False) 
                for j, axs in enumerate(axis):
                    omega = (np.sqrt(Z*curv[axs]/M)/(L*2*np.pi) * 1e-6)/omegas[j][0]
                    loss += (omega - omegas[j][i]/omegas[j][0])**2
            except: 
                sys.exit("The ion in %s positions is lost." %i)
            
    return loss

def v_stoch_grad(uset, stoch, numbers, s, dots, axis, omegas, Z, M, L = 1e-6):
    """
    Calculates stochastic gradient, where partial derivative is calculated 
    only for "stoch:int" randomly chosen parameters

    Returns
    -------
    np.ndarray(numbers[1])
        Calculated gradient

    """
    
    df = np.zeros(numbers[1])
    param = np.random.choice(numbers[1], stoch, replace = 'False')
    param = np.sort(param)
    for i in param:
        uset[i] += 1e-8
        fplus = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L)
        uset[i] -= 2e-8
        fmin = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L)
        uset[i] += 1e-8
        df[i]  = (fplus - fmin)/(2e-8)
        
    return np.array(df)

def v_precise_grad(uset, numbers, s, dots, axis, omegas, Z, M, L = 1e-6):
    """
    Calculates exact gradient of loss function

    Returns
    -------
    np.ndarray(numbers[1])
        Calculated gradient
    """
    
    df = np.zeros(numbers[1])
    for i, el in enumerate(df):
        uset[i] += 1e-8
        fplus = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L)
        uset[i] -= 2e-8
        fmin = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L)
        uset[i] += 1e-8
        df[i]  = (fplus - fmin)/(2e-8)
        
    return np.array(df)



def voltage_optimization(s, numbers, Z, M, dots, axis, omegas, start_uset, learning_rate, stoch = 0, eps = 1e-8, L = 1e-6, step = 100):
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
    numbers : list shape (2)
        [Number of RF electrodes, Number of DC electrodes]
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
    start_uset : list shape (numbers[1])
        Starting voltages for the optimization. The closer they are, the faster the algorithm
    learning_rate : float
        Learning rate of the algorithm. The optimal learning rate may take 
        various values from 0.000001 to 1000. This should be tested in a particular case.
    stoch : int, optional
        The default is 0. This value indicates, if the exact gradient descent 
        (stoch == 0) is performed, or the stochastic gradient descent, with 
        the stochastic choice of "stoch" parameters each iteration.
        The stochastic method is much faster, but for closer voltage sets it
        may loose convergence. In this case, user should switch to the exact gradients.
    eps : float, optional
        Convergence, at each the optimization is stopped. The default is 1e-8.
    L : float, optional
        Length scale in definition of the trap. The default is 1e-6. 
        This means, trap is defined in mkm. 
    step : int, optional
        Number of iterations, at which the current voltage set will be printed.
        The default is 100.

    Returns
    -------
    uset : list shape (numbers[1])
        Resulting voltage set.

    """
    
    uset = start_uset
    loss2 =  v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L)
    print("Initial loss:", loss2)
    
    m = v_precise_grad(start_uset, numbers, s, dots, axis, omegas, Z, M, L)
    v = np.square(m)
    b1 = 0.9
    b2 = 0.999
    big = np.arange(100000)
    small = np.arange(step)
    t = 0
    if stoch == 0:
        for k in big:
            for kk in small:
                grad = v_precise_grad(uset, numbers, s, dots, axis, omegas, Z, M, L)
                m = b1*m+(1-b1)*grad
                v = b2*v+(1-b2)*np.square(grad)
                mtilde = m/(1-b1**(1+t))
                vtilde = v/(1-b2**(1+t))
                v_sqrt = np.sqrt(vtilde) + np.ones(numbers[1])*1e-8
                uset = uset - learning_rate*np.divide(mtilde,v_sqrt)
                t+=1
            loss2 = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L)
            print("Iteration:", t, "Loss function:", loss2)
            print("uset =", list(uset),"\n")
            if loss2 <= eps:
                break
    else:
        for k in big:
            for kk in small:
                grad = v_stoch_grad(uset, stoch, numbers, s, dots, axis, omegas, Z, M, L)
                m = b1*m+(1-b1)*grad
                v = b2*v+(1-b2)*np.square(grad)
                mtilde = m/(1-b1**(1+t))
                vtilde = v/(1-b2**(1+t))
                v_sqrt = np.sqrt(vtilde) + np.ones(numbers[1])*1e-8
                uset = uset - learning_rate*np.divide(mtilde,v_sqrt)
                t+=1
            loss2 = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L)
            print("Iteration:", t, "Loss function:", loss2)
            print("uset =", list(uset),"\n")
            if loss2 <= eps:
                break
    
    return uset
