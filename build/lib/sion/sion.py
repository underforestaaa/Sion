import pylion
from pylion.lammps import lammps
import numpy as np
import matplotlib.pyplot as plt
from electrode import (System, PolygonPixelElectrode, euler_matrix,
                       PointPixelElectrode, PotentialObjective,
                       PatternRangeConstraint, shaped)


"""
Functions, simulating the ion dynamics above planar traps
"""


@lammps.fix
def five_wire_trap(uid, Omega, Urf, u_set, elec, Numb, cover=(0, 0)):
    """
    Simulates simple five-wire planar trap, produced by the FiveWireTrap() function.

    :param uid: str
        ID of ion species, participating in the simulation
    :param Omega: float
        RF frequency of the trap
    :param Urf: float
        Peak voltage of the RF line (in V)
    :param u_set: array shape (Numb+1)
        Set of the voltages on DC electrodes in the same order,
        as specified for FiveWireTrap() function
    :param elec: list
        Coordinates of electrodes in m (obtained from FiveWireTrap())
    :param Numb: int
        number of DC electrodes.
    :param cover: list shape (2)
        array [cover_number, cover_height] - number of the terms in 
        cover electrode influence expansion and its height.
    
    :return: str
        updates simulation lines, which will be executed by LAMMPS
    """
    odict = {}
    lines = [f'\n# Creating a Surface Electrode Trap... (fixID={uid})']
    odict['timestep'] = 1 / Omega / 20
    lines.append(f'variable phase{uid}\t\tequal "{Omega:e}*step*dt"')
    xc = []
    yc = []
    zc = []
    u_set = -u_set

    for iterr in range(2):
        polygon = np.array(elec[0][1][iterr])
        no = np.array(polygon).shape[0]
        for m in range(2 * cover[0] + 1):
            xt = f'(x - ({polygon[no - 1, 0]:e}))'
            numt = no - 1
            yt = f'(y - ({polygon[no - 1, 1]:e}))'
            cov = 2 * (m - cover[0]) * cover[1]
            z = f'(z + ({cov:e}))'

            for k in range(no):
                xo = xt
                yo = yt
                numo = numt
                numt = k
                dx = polygon[numt, 0] - polygon[numo, 0]
                dy = polygon[numo, 1] - polygon[numt, 1]
                c = polygon[numt, 0] * polygon[numo, 1] - \
                    polygon[numo, 0] * polygon[numt, 1]
                lt = (polygon[numt, 0] - polygon[numo, 0]) ** 2 + \
                    (polygon[numt, 1] - polygon[numo, 1]) ** 2

                lines.append(
                    f'variable ro{uid}{iterr:d}{k:d}{m:d} atom "sqrt({xo}^2+{yo}^2+{z}^2)"\n')
                xt = f'(x - ({polygon[k, 0]:e}))'

                yt = f'(y - ({polygon[k, 1]:e}))'
                lines.append(
                    f'variable rt{uid}{iterr:d}{k:d}{m:d} atom "sqrt({xt}^2+{yt}^2+{z}^2)"\n')
                lines.append(
                    f'variable n{uid}{iterr:d}{k:d}{m:d} atom "{Urf:e}*(v_ro{uid}{iterr:d}{k:d}{m:d}+v_rt{uid}{iterr:d}{k:d}{m:d})/(v_ro{uid}{iterr:d}{k:d}{m:d}*v_rt{uid}{iterr:d}{k:d}{m:d}*((v_ro{uid}{iterr:d}{k:d}{m:d}+v_rt{uid}{iterr:d}{k:d}{m:d})*(v_ro{uid}{iterr:d}{k:d}{m:d}+v_rt{uid}{iterr:d}{k:d}{m:d})-({lt:e}))*{np.pi:e})"\n')

                if dx == 0:
                    yc.append(f'0')
                else:
                    yc.append(
                        f'({dx:e})*{z}*v_n{uid}{iterr:d}{k:d}{m:d}*cos(v_phase{uid})')
                if dy == 0:
                    xc.append(f'0')
                else:
                    xc.append(
                        f'({dy:e})*{z}*v_n{uid}{iterr:d}{k:d}{m:d}*cos(v_phase{uid})')
                zc.append(
                    f'({c:e} - ({dx:e})*y - ({dy:e})*x)*v_n{uid}{iterr:d}{k:d}{m:d}*cos(v_phase{uid})')
    xc = ' + '.join(xc)
    yc = ' + '.join(yc)
    zc = ' + '.join(zc)

    xr = []
    yr = []
    zr = []
    for ite in range(Numb):
        polygon = np.array(elec[ite+1][1][0])

        no = np.array(polygon).shape[0]
        for m in range(2 * cover[0] + 1):
            x2 = f'(x - ({polygon[no - 1, 0]:e}))'
            y2 = f'(y - ({polygon[no - 1, 1]:e}))'
            numt = no - 1
            cov = 2 * (m - cover[0]) * cover[1]
            z = f'(z + ({cov:e}))'

            for k in range(no):
                x1 = x2
                y1 = y2
                numo = numt
                numt = k
                lines.append(
                    f'variable rodc{uid}{ite:d}{k:d}{m:d} atom "sqrt({x2}^2+{y2}^2+{z}^2)"\n')
                x2 = f'(x - ({polygon[k, 0]:e}))'
                y2 = f'(y - ({polygon[k, 1]:e}))'
                lines.append(
                    f'variable rtdc{uid}{ite:d}{k:d}{m:d} atom "sqrt({x2}^2+{y2}^2+{z}^2)"\n')
                dx = polygon[numt, 0] - polygon[numo, 0]
                dy = polygon[numo, 1] - polygon[numt, 1]
                c = polygon[numt, 0] * polygon[numo, 1] - \
                    polygon[numo, 0] * polygon[numt, 1]
                lt = (polygon[numt, 0] - polygon[numo, 0]) ** 2 + \
                    (polygon[numt, 1] - polygon[numo, 1]) ** 2
                lines.append(
                    f'variable ndc{uid}{ite:d}{k:d}{m:d} atom "{u_set[ite+1]:e}*(v_rodc{uid}{ite:d}{k:d}{m:d}+v_rtdc{uid}{ite:d}{k:d}{m:d})/(v_rodc{uid}{ite:d}{k:d}{m:d}*v_rtdc{uid}{ite:d}{k:d}{m:d}*((v_rodc{uid}{ite:d}{k:d}{m:d}+v_rtdc{uid}{ite:d}{k:d}{m:d})*(v_rodc{uid}{ite:d}{k:d}{m:d}+v_rtdc{uid}{ite:d}{k:d}{m:d})-({lt:e}))*{np.pi:e})"\n')

                if dy == 0:
                    xr.append(f'0')
                else:
                    xr.append(f'({dy:e})*{z}*v_ndc{uid}{ite:d}{k:d}{m:d}')
                if dx == 0:
                    yr.append(f'0')
                else:
                    yr.append(f'({dx:e})*{z}*v_ndc{uid}{ite:d}{k:d}{m:d}')
                zr.append(
                    f'({c:e} - ({dx:e})*y - ({dy:e})*x)*v_ndc{uid}{ite:d}{k:d}{m:d}')
    xr = ' + '.join(xr)
    yr = ' + '.join(yr)
    zr = ' + '.join(zr)

    lines.append(f'variable oscEX{uid} atom "{xc}+{xr}"')
    lines.append(f'variable oscEY{uid} atom "{yc}+{yr}"')
    lines.append(f'variable oscEZ{uid} atom "{zc}+{zr}"')
    lines.append(
        f'fix {uid} all efield v_oscEX{uid} v_oscEY{uid} v_oscEZ{uid}\n')

    odict.update({'code': lines})

    return odict


@lammps.fix
def polygon_trap(uid, Omega, u_set, RFs, DCs, cover=(0, 0)):
    """
    Simulates an arbitrary planar trap with polygonal electrodes

    :param uid: str
        ions ID
    :param Omega: float
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
    xc = []
    yc = []
    zc = []
    u_set = -u_set

    for iterr in range(nrf):
        polygon = np.array(RFs[iterr])
        no = np.array(polygon).shape[0]

        for m in range(2*cover[0]+1):
            xt = f'(x - ({polygon[no-1, 0]:e}))'
            numt = no-1
            yt = f'(y - ({polygon[no-1, 1]:e}))'
            cov = 2*(m - cover[0])*cover[1]
            z = f'(z + ({cov:e}))'

            for k in range(no):
                xo = xt
                yo = yt
                numo = numt
                numt = k
                dx = polygon[numt, 0] - polygon[numo, 0]
                dy = polygon[numo, 1] - polygon[numt, 1]
                c = polygon[numt, 0] * polygon[numo, 1] - \
                    polygon[numo, 0] * polygon[numt, 1]
                lt = (polygon[numt, 0] - polygon[numo, 0]) ** 2 + \
                    (polygon[numt, 1] - polygon[numo, 1]) ** 2

                lines.append(
                    f'variable ro{uid}{iterr:d}{k:d}{m:d} atom "sqrt({xo}^2+{yo}^2+{z}^2)"\n')
                xt = f'(x - ({polygon[k, 0]:e}))'

                yt = f'(y - ({polygon[k, 1]:e}))'
                lines.append(
                    f'variable rt{uid}{iterr:d}{k:d}{m:d} atom "sqrt({xt}^2+{yt}^2+{z}^2)"\n')
                lines.append(
                    f'variable n{uid}{iterr:d}{k:d}{m:d} atom "({u_set[iterr]:e})*(v_ro{uid}{iterr:d}{k:d}{m:d}+v_rt{uid}{iterr:d}{k:d}{m:d})/(v_ro{uid}{iterr:d}{k:d}{m:d}*v_rt{uid}{iterr:d}{k:d}{m:d}*((v_ro{uid}{iterr:d}{k:d}{m:d}+v_rt{uid}{iterr:d}{k:d}{m:d})*(v_ro{uid}{iterr:d}{k:d}{m:d}+v_rt{uid}{iterr:d}{k:d}{m:d})-({lt:e}))*{np.pi:e})"\n')

                if dx == 0:
                    yc.append(f'0')
                else:
                    yc.append(
                        f'({dx:e})*{z}*v_n{uid}{iterr:d}{k:d}{m:d}*cos(v_phase{uid}{iterr:d})')
                if dy == 0:
                    xc.append(f'0')
                else:
                    xc.append(
                        f'({dy:e})*{z}*v_n{uid}{iterr:d}{k:d}{m:d}*cos(v_phase{uid}{iterr:d})')
                zc.append(
                    f'({c:e} - ({dx:e})*y - ({dy:e})*x)*v_n{uid}{iterr:d}{k:d}{m:d}*cos(v_phase{uid}{iterr:d})')
    xc = ' + '.join(xc)
    yc = ' + '.join(yc)
    zc = ' + '.join(zc)

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
                lines.append(
                    f'variable rodc{uid}{ite:d}{k:d}{m:d} atom "sqrt({x2}^2+{y2}^2+{z}^2)"\n')
                x2 = f'(x - ({polygon[k, 0]:e}))'
                y2 = f'(y - ({polygon[k, 1]:e}))'
                lines.append(
                    f'variable rtdc{uid}{ite:d}{k:d}{m:d} atom "sqrt({x2}^2+{y2}^2+{z}^2)"\n')
                dx = polygon[numt, 0] - polygon[numo, 0]
                dy = polygon[numo, 1] - polygon[numt, 1]
                c = polygon[numt, 0] * polygon[numo, 1] - \
                    polygon[numo, 0] * polygon[numt, 1]
                lt = (polygon[numt, 0] - polygon[numo, 0]) ** 2 + \
                    (polygon[numt, 1] - polygon[numo, 1]) ** 2
                lines.append(
                    f'variable ndc{uid}{ite:d}{k:d}{m:d} atom "({u_set[ite+nrf]:e})*(v_rodc{uid}{ite:d}{k:d}{m:d}+v_rtdc{uid}{ite:d}{k:d}{m:d})/(v_rodc{uid}{ite:d}{k:d}{m:d}*v_rtdc{uid}{ite:d}{k:d}{m:d}*((v_rodc{uid}{ite:d}{k:d}{m:d}+v_rtdc{uid}{ite:d}{k:d}{m:d})*(v_rodc{uid}{ite:d}{k:d}{m:d}+v_rtdc{uid}{ite:d}{k:d}{m:d})-({lt:e}))*{np.pi:e})"\n')

                if dy == 0:
                    xr.append(f'0')
                else:
                    xr.append(f'({dy:e})*{z}*v_ndc{uid}{ite:d}{k:d}{m:d}')
                if dx == 0:
                    yr.append(f'0')
                else:
                    yr.append(f'({dx:e})*{z}*v_ndc{uid}{ite:d}{k:d}{m:d}')
                zr.append(
                    f'({c:e} - ({dx:e})*y - ({dy:e})*x)*v_ndc{uid}{ite:d}{k:d}{m:d}')
    xr = ' + '.join(xr)
    yr = ' + '.join(yr)
    zr = ' + '.join(zr)

    lines.append(f'variable oscEX{uid} atom "{xc}+{xr}"')
    lines.append(f'variable oscEY{uid} atom "{yc}+{yr}"')
    lines.append(f'variable oscEZ{uid} atom "{zc}+{zr}"')
    lines.append(
        f'fix {uid} all efield v_oscEX{uid} v_oscEY{uid} v_oscEZ{uid}\n')

    odict.update({'code': lines})

    return odict


@lammps.fix
def point_trap(uid, Omega, trap, voltages, cover=(0, 0)):
    """
    Simulates an arbitrary point trap. The point trap means a trap of 
    an arbitrary shape, which is approximated by circle-shaped electrodes, 
    called points. The points potential is approximated from the fact, that it 
    has infinitesemal radius, so the smaller are points, the more precise is
    the simulation (but slower).

    :param: Omega: float
        array of RF frequencies of all RF points.
    :param: trap: list shape (4, shape(element))
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
    odict['timestep'] = 1 / np.max(Omega) / 20
    nrf = len(trap[0])
    ndc = len(trap[2])
    RF_areas = trap[0]
    RF_coord = trap[1]
    DC_areas = trap[2]
    DC_coord = trap[3]
    for i in range(nrf):
        lines.append(
            f'variable phase{uid}{i:d}\t\tequal "{Omega[i]:e}*step*dt"')
    xr = []
    yr = []
    zr = []

    for i in range(nrf):
        for m in range(2*cover[0]+1):
            xt = f'(x - ({RF_coord[i][0]:e}))'
            yt = f'(y - ({RF_coord[i][1]:e}))'
            cov = 2 * (m - cover[0]) * cover[1]
            z = f'(z + ({cov:e}))'

            lines.append(
                f'variable r{uid}{i:d}{m:d} atom "sqrt({xt}^2+{yt}^2+{z}^2)"\n')

            lines.append(
                f'variable n{uid}{i:d}{m:d} atom "( {RF_areas[i]*voltages[0][i]:e}/(2*{np.pi:e}*v_r{uid}{i:d}{m:d}^5) )"\n')

            xr.append(
                f'(-3*{xt}*{z}*v_n{uid}{i:d}{m:d})*cos(v_phase{uid}{i:d})')
            yr.append(
                f'(-3*{yt}*{z}*v_n{uid}{i:d}{m:d})*cos(v_phase{uid}{i:d})')
            zr.append(
                f'((v_r{uid}{i:d}{m:d}^2 - 3*{z}^2)*v_n{uid}{i:d}{m:d})*cos(v_phase{uid}{i:d})')

    xr = ' + '.join(xr)
    yr = ' + '.join(yr)
    zr = ' + '.join(zr)

    xc = []
    yc = []
    zc = []

    for i in range(ndc):
        for m in range(2*cover[0]+1):
            xt = f'(x - ({DC_coord[i][0]:e}))'
            yt = f'(y - ({DC_coord[i][1]:e}))'
            cov = 2 * (m - cover[0]) * cover[1]
            z = f'(z + ({cov:e}))'

            lines.append(
                f'variable r{uid}{i:d}{m:d} atom "sqrt({xt}^2+{yt}^2+{z}^2)"\n')

            lines.append(
                f'variable n{uid}{i:d}{m:d} atom "( {DC_areas[i]*voltages[1][i]:e}/(2*{np.pi:e}*v_r{uid}{i:d}{m:d}^5) )"\n')

            xc.append(f'(-3*{xt}*{z}*v_n{uid}{i:d}{m:d})')
            yc.append(f'(-3*{yt}*{z}*v_n{uid}{i:d}{m:d})')
            zc.append(f'((v_r{uid}{i:d}{m:d}^2 - 3*{z}^2)*v_n{uid}{i:d}{m:d})')

    xc = ' + '.join(xc)
    yc = ' + '.join(yc)
    zc = ' + '.join(zc)

    lines.append(f'variable oscEX{uid} atom "{xr}+{xc}"')
    lines.append(f'variable oscEY{uid} atom "{yr}+{yc}"')
    lines.append(f'variable oscEZ{uid} atom "{zr}+{zc}"')
    lines.append(
        f'fix {uid} all efield v_oscEX{uid} v_oscEY{uid} v_oscEZ{uid}\n')

    odict.update({'code': lines})

    return odict


@lammps.fix
def ringtrap(uid, Omega, r, R, voltage, resolution=100, center=(0, 0), cover=(0, 0)):
    """
    Simulates a ring-shaped RF electrode

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
        updates simulation of one single ring

    """
    x = []
    a = []
    count = 0
    rn = r
    while (rn < R):
        rl = rn
        rn = (resolution + np.pi) * rn / (resolution - np.pi)
        r0 = (rn - rl) / 2
        for i in range(resolution):
            phi = 2 * np.pi * i / resolution
            ksi = np.array([(rl + r0) * np.cos(phi), (rl + r0)
                           * np.sin(phi)]) + np.array(center)
            x.append(ksi)
            a.append(np.pi * r0 ** 2)
        count += 1

    omega = np.ones(count*resolution)*Omega
    voltages = np.ones(count*resolution)*voltage
    volt = [voltages, [0]]

    return point_trap(omega, [a, x, [0], [[0, 0]]], volt, cover)


"""
Definition of particularly useful planar traps
"""


def FiveWireTrap(Urf, DCtop, DCbottom, cwidth, clength, boardwidth, rftop, rflength, rfbottom, patternTop=None,
                 patternBot=None, getCoordinate=None, gapped=None, cheight=0, cmax=0, plott=None):
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
    if not gapped:
        gapped = 0

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
    if patternTop is None:
        pt = 1
    else:
        pt = patternTop
    DCtop = DCtop * pt
    DCtop = np.array(DCtop)

    if patternBot is None:
        pb = 1
    else:
        pb = patternBot
    DCb = DCb * pb
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
        ("rf", [rf_top,
                rf_bottom])]
    for i in range(n):
        st = "t[" + str(i % (n // pt) + 1) + "]"
        electrodes.append([st, [t[i]]])
    for i in range(nb):
        st = "b[" + str(i % (nb // pb) + 1) + "]"
        electrodes.append([st, [b[i]]])
    electrodes.append(["c", [c]])

    # Polygon approach. All DCs are 0 for nown
    s = System([PolygonPixelElectrode(cover_height=cheight, cover_nmax=cmax, name=n, paths=map(np.array, p))
                for n, p in electrodes])
    s["rf"].rf = Urf

    for i in range(n):
        st = "t[" + str(i % (n // pt) + 1) + "]"
        s[st].dc = 0
    for i in range(nb):
        st = "b[" + str(i % (nb // pb) + 1) + "]"
        s[st].dc = 0
    s["c"].dc = 0

    # Exact coordinates for lion
    elec = [
        ("rf", [np.array(rf_top) * 1e-6,
                np.array(rf_bottom) * 1e-6])]
    for i in range(n):
        st = "t[" + str(i % (n // pt) + 1) + "]"
        elec.append([st, [np.array(t[i]) * 1e-6]])
    for i in range(nb):
        st = "b[" + str(i % (nb // pb) + 1) + "]"
        elec.append([st, [np.array(b[i]) * 1e-6]])
    elec.append(["c", [np.array(c) * 1e-6]])

    # Part of getting coordinates of electrodes
    if getCoordinate is not None:

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
            ("rf", [rf_top,
                    rf_bottom])]
        for i in range(n):
            st = "t[" + str(i % (n // pt) + 1) + "]"
            coordinates.append([st, [t[i]]])
        for i in range(nb):
            st = "b[" + str(i % (nb // pb) + 1) + "]"
            coordinates.append([st, [b[i]]])
        coordinates.append(["c", [c]])

        # Writes them in a file
        with open('coordinates.txt', 'w') as f:
            for item in coordinates:
                f.write(f'{item}\n')

    # creates a plot of electrode
    if plott is not None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))
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

    return elec, n + nb + 1, s


def linearrings(r, R, res, dif, Nring, rdc=None):
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
    x = []
    a = []
    countrf = 0
    for ket in range(Nring):
        rn = r
        while (rn < R):
            rl = rn
            rn = (res + np.pi) * rn / (res - np.pi)
            r0 = (rn - rl) / 2
            for i in range(res):
                phi = 2 * np.pi * i / res
                ksi = np.array([(rl + r0) * np.cos(phi), (rl + r0)
                               * np.sin(phi)]) + np.array([dif * ket, 0])
                x.append(ksi)
                a.append(np.pi * r0 ** 2)
            countrf += 1
    
    if rdc:
        countdc = 0
        cc = 0
        resdc = 20
        for ket in range(Nring):
            rn = rdc/10
            while (rn < rdc):
                rl = rn
                rn = (resdc + np.pi) * rn / (resdc - np.pi)
                r0 = (rn - rl) / 2
                for i in range(resdc):
                    phi = 2 * np.pi * i / resdc
                    ksi = np.array([(rl + r0) * np.cos(phi), (rl + r0)
                                   * np.sin(phi)]) + np.array([dif * ket, 0])
                    x.append(ksi)
                    a.append(np.pi * r0 ** 2)
                countdc += 1

        for gr in range(Nring):
            x.append([gr * dif, 0])
            a.append(np.pi * (rdc/10) ** 2)
            cc += 1
    return x, a, [countrf*res, countdc*resdc+cc]



def n_rf_trap(Urf, DCtop, DCbottom, cwidth, rfwidth, rflength, n_rf=1, patternTop=None,
              patternBot=None, cheight=0, cmax=0, plott=None):
    """
    A function for convenient definition of multi-wire traps, with n symmetrical RF lines.

    :param Urf: float
        A parametrized RF voltage for pseudopotential approximation (see examples)
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
    :param rfwidth: float
        width of an RF electrode, forming the RF lines
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
    :return: N: int
        number of all DC electrodes
    :return: s: electrode.System object
        system object for electrode package.
    """

    c = [[-rflength/2, 0], [rflength/2, 0],
         [rflength/2, cwidth], [-rflength/2, cwidth]]
    RF = []
    electrodes = []
    for i in range(n_rf):
        rf_top = [[-rflength/2, cwidth+i*rfwidth], [rflength/2, cwidth+i*rfwidth],
                  [rflength/2, cwidth+(i+1)*rfwidth], [-rflength/2, cwidth+(i+1)*rfwidth]]
        rf_bottom = [[-rflength / 2,  -(i+1) * rfwidth], [rflength / 2, -(i+1) * rfwidth],
                     [rflength / 2, -i * rfwidth], [-rflength / 2, -i * rfwidth]]
        st = 'rf' + str(i)
        electrodes.append([st, [rf_top, rf_bottom]])
        RF.append(np.array(rf_top)*1e-6)
        RF.append(np.array(rf_bottom) * 1e-6)
    # define arrays of DCs considering pattern
    if patternTop is None:
        pt = 1
    else:
        pt = patternTop
    DCtop = DCtop * pt
    DCtop = np.array(DCtop)

    if patternBot is None:
        pb = 1
    else:
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

    DC = []
    # starting point - central electrode, among top DCs. Considering parity of electrode number (i just want it to be BEAUTIFUL)
    if n % 2 == 0:
        t[m] = np.array([[0, rfwidth*n_rf+cwidth + DCtop[m][0]],
                         [DCtop[m][1], rfwidth*n_rf+cwidth + DCtop[m][0]],
                         [DCtop[m][1], rfwidth*n_rf+cwidth],
                         [0, rfwidth*n_rf+cwidth]])
    else:
        t[m] = np.array([[- DCtop[m][1] / 2, rfwidth*n_rf+cwidth + DCtop[m][0]],
                         [+ DCtop[m][1] / 2, rfwidth*n_rf+cwidth + DCtop[m][0]],
                         [+ DCtop[m][1] / 2, rfwidth*n_rf+cwidth],
                         [- DCtop[m][1] / 2, rfwidth*n_rf+cwidth]])

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
            [[0, -rfwidth*n_rf], [+ DCb[m][1], -rfwidth*n_rf],
             [+ DCb[m][1], -rfwidth*n_rf - DCb[m][0]],
             [0, -rfwidth*n_rf - DCb[m][0]]])
    else:
        b[m] = np.array([[- DCb[m][1] / 2, -rfwidth*n_rf],
                         [+ DCb[m][1] / 2, -rfwidth*n_rf],
                         [+ DCb[m][1] / 2, -rfwidth*n_rf - DCb[m][0]],
                         [- DCb[m][1] / 2, -rfwidth*n_rf - DCb[m][0]]])

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
        st = "t[" + str(i % (n // pt) + 1) + "]"
        electrodes.append([st, [t[i]]])
    for i in range(nb):
        st = "b[" + str(i % (nb // pb) + 1) + "]"
        electrodes.append([st, [b[i]]])
    electrodes.append(["c", [c]])

    # Polygon approach. All DCs are 0 for nown
    s = System([PolygonPixelElectrode(cover_height=cheight, cover_nmax=cmax, name=n, paths=map(np.array, p))
                for n, p in electrodes])
    for i in range(n_rf):
        st = 'rf' + str(i)
        s[st].rf = Urf[i]

    for i in range(n):
        st = "t[" + str(i % (n // pt) + 1) + "]"
        s[st].dc = 0
    for i in range(nb):
        st = "b[" + str(i % (nb // pb) + 1) + "]"
        s[st].dc = 0
    s["c"].dc = 0
    DC.extend(np.array(t)*1e-6)
    DC.extend(np.array(b)*1e-6)
    # Exact coordinates for lion
    elec = [RF, DC]

    # creates a plot of electrode
    if plott is not None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))
        s.plot(ax[0])
        s.plot_voltages(ax[1], u=s.rfs)
        # u = s.rfs sets the voltage-type for the voltage plot to RF-voltages (DC are not shown)
        xmax = rflength * 2 / 3
        ymaxp = (np.max(DCtop) + rfwidth*n_rf + cwidth) * 1.2
        ymaxn = (np.max(DCbottom) + rfwidth*n_rf) * 1.2
        ax[0].set_title("colour")
        # ax[0] addresses the first plot in the subplots - set_title gives this plot a title
        ax[1].set_title("rf-voltages")
        for axi in ax.flat:
            axi.set_aspect("equal")
            axi.set_xlim(-xmax, xmax)
            axi.set_ylim(-ymaxn, ymaxp)

    return elec, n + nb + 1, s


def individual_wells(Urf, width, length, dot, x, y):
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
    
    :return: s: electrode.System object
        System object for electrode package simulation
    :return: electrodes: list shape (2, shape of electrodes)
        coordinates of RF electrodes for Sion simulation
    """
    rfstart = [[-length/2, -width/2], [dot[0,0]-x[0]/2, -width/2], [dot[0,0]-x[0]/2, width/2], [-length/2, width/2]]
    rfs = [rfstart]
    n = dot.shape[0]
    
    for i in range(n-1):
        rfs.append([[dot[i, 0]-x[i]/2, dot[i, 1]+y[i]/2], [dot[i, 0]+x[i]/2, dot[i, 1]+y[i]/2], [dot[i, 0]+x[i]/2, width/2], [dot[i, 0]-x[i]/2, width/2]])
        rfs.append([[dot[i, 0]-x[i]/2, -width/2],[dot[i, 0]+x[i]/2, -width/2], [dot[i, 0]+x[i]/2, dot[i, 1]-y[i]/2], [dot[i, 0]-x[i]/2, dot[i, 1]-y[i]/2]])
        rfs.append([[dot[i,0]+x[i], -width/2], [dot[i+1,0]-x[i], -width/2], [dot[i+1,0]-x[i], width/2],[dot[i,0]+x[i], width/2]])
    
    i = n-1
      
    rfs.append([[dot[i, 0]-x[i]/2, dot[i, 1]+y[i]/2], [dot[i, 0]+x[i]/2, dot[i, 1]+y[i]/2], [dot[i, 0]+x[i]/2, width/2], [dot[i, 0]-x[i]/2, width/2]])
    rfs.append([[dot[i, 0]-x[i]/2, -width/2],[dot[i, 0]+x[i]/2, -width/2], [dot[i, 0]+x[i]/2, dot[i, 1]-y[i]/2], [dot[i, 0]-x[i]/2, dot[i, 1]-y[i]/2]])
    rfs.append([[dot[i,0]+x[i], -width/2], [length/2, -width/2], [length/2, width/2],[dot[i,0]+x[i], width/2]])
    

    electrodes = [
        ("+rf", rfs)]

    
    s = System([PolygonPixelElectrode(name=f, paths=map(np.array, p))
                for f, p in electrodes])
    
    return s, [electrodes[0][1], [[0,0,0,0]]]




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

ech = 1.602176634e-19  # electron charge, C
amu = 1.66053906660e-27  # atomic mass unit, kg
eps0 = 8.8541878128e-12  # vacuum electric permittivity


def axial_linear_hessian_matrix(ion_positions, omega_sec, Z, mass):
    """
    Calculates axial hessian and M_matrix (x-ax) of a linear ion chain

    :param: ion_positions: np.ndarray [ion_number, 3]
        array of final ion positions
    :param: omega_sec: float
        axial secular frequency of the trap in potential minimum
    :param: Z: int
        ion charge in elementary charge units
    :param: mass: ion mass in kg
    
    :return: list (A_matrix, M_matrix): 
        A_matrix is axial hessian matrix, M - matrix of ion masses

    """
    global amu, ech, eps0
    Z = Z*ech
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
    :param: Z: int
        ion charge in elementary charge units
    :param: mass: float
        ion mass in kg
    
    :return: list (B_matrix, M_matrix)
        B_matrix is hessian matrix.
        """
    global amu, ech, eps0
    Z = Z*ech
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
    :param: Z: int
        ion charge in elemetary charge units
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
    :param: Z: int
        ion charge in elemetary charge units
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
        array of anharmonic scale lengths in each calculated potential minimum. 
        Given as  [l2, l3, l4], for ln being scale length of the n's potential term

    """
    
    results = []
    for pos in minimums:
        x1 = s.minimum(np.array(pos), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
        curv_z, mod_dir=s.modes(x1,sorted=False)
        curv_z *= L**2
        inv = np.linalg.inv(mod_dir)
        pot = s.potential(x1, derivative = 4)[0,axis,axis]
        eig_pot = np.dot(inv, np.dot(pot, mod_dir))
        eig = eig_pot[axis,axis]*L**4
        k2 = curv_z[axis]*ech
        k4 = eig*ech
        pot3 = s.potential(x1, derivative = 3)[0,axis]
        eig_pot3 = np.dot(inv, np.dot(pot3, mod_dir))
        eig3 = eig_pot3[axis,axis]*L**3
        k3 = eig3*ech
        l2 = (ech**2/(4*np.pi*eps0*k2))**(1/3)
        l3 = l2*(k3/k2)**(1/(2-3))
        l4 = l2*(k4/k2)**(1/(2-4))
        results.append([l2, l3, l4])
    
    return results


def anharmonic_axial_modes(s, ion_positions, masses, axis, L = 1e-6):
    """
    Anharmonic normal modes along the chosen pricniple axis of oscillation
    in a linear ion chain,
    considering hexapole and octopole anharmonicities.
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
        x1 = s.minimum(np.array(pos), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
        minims.append(x1)
        curv_z, mod_dir=s.modes(x1,sorted=False)
        curv_z *= L**2
        inv = np.linalg.inv(mod_dir)
        pot = s.potential(x1, derivative = 4)[0,axis,axis]
        eig_pot = np.dot(inv, np.dot(pot, mod_dir))
        eig = eig_pot[axis,axis]*L**4
        k2 = curv_z[axis]*ech
        k4 = eig*ech
        pot3 = s.potential(x1, derivative = 3)[0,axis]
        eig_pot3 = np.dot(inv, np.dot(pot3, mod_dir))
        eig3 = eig_pot3[axis,axis]*L**3
        k3 = eig3*ech
        l = (ech**2/(4*np.pi*eps0*k2))**(1/3)
        alpha2.append(k2)
        alpha3.append(l*k3/k2)
        alpha4.append(l**2*k4/k2)
    
    lo = (ech**2/(4*np.pi*eps0*alpha2[0]))**(1/3)
    ion_positions = np.array(ion_positions)/lo
    minims = np.array(minims)/lo
    hessian = np.zeros([n,n])
    for i in range(n):
        hessian[i,i] = alpha2[i]/alpha2[0] + 2*alpha3[i]/(alpha2[i]/alpha2[0])*np.abs(ion_positions[i]-minims[i]) + 6*alpha4[i]*(alpha2[i]/alpha2[0])*(ion_positions[i]-minims[i])**2
        coulomb = 0
        for p in range(n):
            if p != i:
                coulomb += B[axis][0]/(np.abs(ion_positions[i]-ion_positions[p]))**3
        hessian[i,i] += coulomb
    for i in range(n):
        for j in range(n):
            if i!=j:
                hessian[i,j] = B[axis][1]/(np.abs(ion_positions[i]-ion_positions[j]))**3
    for i, m in enumerate(masses):
        masses[i] = 1/m**0.5
    M_matrix = np.diag(masses)
    eig, vec = np.linalg.eig(np.dot(M_matrix, np.dot(hessian, M_matrix)))
    
    return np.sqrt(eig*alpha2[0])/(2*np.pi), vec


"""
Stability analysis
"""

def stability(s, M, Omega, Z, dot, L = 1e-6, plot = 1):
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
    params : list [areal, qreal, thetha, [a_crit_upper, a_crit_lower], q_crit]
        Stability parameters for this trap, sorted as follows:
            areal: calculated a parameter for this trap and voltage set
            qreal: calculated q parameter for this trap and voltage set
            thetha: angle of rotation between RF nad DC potential axes
            a_crit_upper: maximal achievable positive a parameter in this trap
            a_crit_lower: maximal achievable negative a parameter in this trap
            q_crit: maximal achievable q parameter in this trap
    plt.plot : optional
        plot of the stability diagram and trap's a-q parameters on it 

    """
    
    scale = Z/((L*Omega)**2*M)
    params = []
    
    x1 = s.minimum(np.array(dot), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
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
    params.append(areal)

    #diagonalize Q
    DiagQ, BasQ = np.linalg.eig(Q)
    DiagQ = np.diag(DiagQ)
    qreal = DiagQ[0,0]
    params.append(qreal)

    #obtain theta
    rot = np.dot(np.linalg.inv(BasQ),BasA)
    thetha = np.arccos(rot[0,0])
    c = np.cos(2*thetha)
    sin = np.sin(2*thetha)
    params.append(thetha)

    #boundaries
    q = np.linspace(0,1.5, 1000)
    aa = -q**2/2
    ab = q**2/(2*alpha)
    ac = 1 - c*q - (c**2/8 + (2*sin**2*(5+alpha))/((1+alpha)*(9+alpha)))*q**2
    ad = -(1 - c*q - (c**2/8 + (2*sin**2*(5+1/alpha))/((1+1/alpha)*(9+1/alpha)))*q**2)/alpha
    
    #critical a, q
    aa = 1/(2*alpha)
    ee = 1
    ff = -c
    gg = -(c**2/8 + (2*sin**2*(5+alpha))/((1+alpha)*(9+alpha)))
    bb = -1/alpha
    cc = c/alpha
    dd = (c**2/8 + (2*sin**2*(5+1/alpha))/((1+1/alpha)*(9+1/alpha)))/alpha
    hh = -1/2
    a_crit_upper = np.min(np.abs(np.array(([(-cc - np.sqrt(cc**2 - 4*bb*(dd-aa)))/(2*(dd-aa)), (-cc + np.sqrt(cc**2 - 4*bb*(dd-aa)))/(2*(dd-aa))]))))
    a_crit_lower = -np.min(np.abs(np.array(([(-ff - np.sqrt(ff**2 - 4*ee*(gg-hh)))/(2*(gg-hh)), (-ff + np.sqrt(ff**2 - 4*ee*(gg-hh)))/(2*(gg-hh))]))))
    
    params.append([a_crit_upper, a_crit_lower])
    q_crit = np.min([(-(cc-ff) - np.sqrt((cc-ff)**2 - 4*(dd-gg)*(bb-ee)))/(2*(dd-gg)), (-(cc-ff) + np.sqrt((cc-ff)**2 - 4*(dd-gg)*(bb-ee)))/(2*(dd-gg))])
    params.append(q_crit)
    
    if plot == 1:
        #plotting boundaries 
        fig = plt.figure()
        fig.set_size_inches(7,5)
        plt.plot(q,aa, 'g')
        plt.plot(q,ab, 'g')
        plt.plot(q,ac, 'g')
        plt.plot(q,ad, 'g')
        
        #unifying noudaries to two united line so the area between them can be filled by plt.fill_between
        y1 = np.array(list(map(max, zip(aa, ad))))
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
        plt.plot(qreal, areal, 'ro', markersize= 8)
        
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
    try: 
        with s.with_voltages(dcs = u_set, rfs = None):
            for i, pos in enumerate(dots):
                x1 = s.minimum(pos, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
                curv, mod_dir=s.modes(x1,sorted=False) 
                for j, axs in enumerate(axis):
                    omega = (np.sqrt(Z*curv[axs]/M)/(L*2*np.pi) * 1e-6)/omegas[j][0]
                    loss += (omega - omegas[j][i]/omegas[j][0])**2
    except: 
        loss = 1000
            
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
    if stoch == 0:
        for t in big:
            for kk in small:
                grad = v_precise_grad(uset, numbers, s, dots, axis, omegas, Z, M, L)
                m = b1*m+(1-b1)*grad
                v = b2*v+(1-b2)*np.square(grad)
                mtilde = m/(1-b1**(1+t))
                vtilde = v/(1-b2**(1+t))
                v_sqrt = np.sqrt(vtilde) + np.ones(numbers[1])*1e-8
                uset = uset - learning_rate*np.divide(mtilde,v_sqrt)
            loss2 = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L)
            print("Iteration:", t+1, "Loss function:", loss2)
            print("uset =", list(uset),"\n")
            if loss2 <= eps:
                break
    else:
        for t in big:
            for kk in small:
                grad = v_stoch_grad(uset, stoch, numbers, s, dots, axis, omegas, Z, M, L)
                m = b1*m+(1-b1)*grad
                v = b2*v+(1-b2)*np.square(grad)
                mtilde = m/(1-b1**(1+t))
                vtilde = v/(1-b2**(1+t))
                v_sqrt = np.sqrt(vtilde) + np.ones(numbers[1])*1e-8
                uset = uset - learning_rate*np.divide(mtilde,v_sqrt)
            loss2 = v_lossf(uset, numbers, s, dots, axis, omegas, Z, M, L)
            print("Iteration:", t+1, "Loss function:", loss2)
            print("uset =", list(uset),"\n")
            if loss2 <= eps:
                break
    
    return uset




def g_lossf(geom, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L = 1e-6):
    """
    Loss function for the geometry optimization.
    If the individual potential well is lost, returns 1000
    
    Returns
    -------
    loss : float
        Loss value, calculated as a quadratic norm of difference between the 
        actual and desired set of secular frequencies

    """

    loss = 0
    x = geom[0:n_dots]
    y = geom[n_dots:2*n_dots]
    
    #obtain result of the function
    try: 
        s = individual_wells(Urf, width, length, dots, x, y)
        for i, pos in enumerate(dots):
            x1 = s.minimum(pos, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
            curv, mod_dir=s.modes(x1,sorted=False) 
            for j, axs in enumerate(axis):
                omega = (np.sqrt(Z*curv[axs]/M)/(L*2*np.pi) * 1e-6)/omegas[j][0]
                loss += (omega - omegas[j][i]/omegas[j][0])**2
    except: 
        loss = 1000
            
    return loss

def g_stoch_grad(geom, stoch, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L = 1e-6):
    """
    Calculates stochastic gradient, where partial derivative is calculated 
    only for "stoch:int" randomly chosen parameters

    Returns
    -------
    np.ndarray(n_dots)
        Calculated gradient

    """
    
    df = np.zeros(n_dots)
    param = np.random.choice(n_dots, stoch, replace = 'False')
    param = np.sort(param)
    for i in param:
        geom[i] += 1e-8
        fplus = g_lossf(geom, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L)
        geom[i] -= 2e-8
        fmin = g_lossf(geom, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L)
        geom[i] += 1e-8
        df[i]  = (fplus - fmin)/(2e-8)
        
    return np.array(df)

def g_precise_grad(geom, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L = 1e-6):
    """
    Calculates exact gradient of loss function

    Returns
    -------
    np.ndarray(n_dots)
        Calculated gradient
    """
    
    df = np.zeros(n_dots)
    for i, el in enumerate(df):
        geom[i] += 1e-8
        fplus = g_lossf(geom, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L)
        geom[i] -= 2e-8
        fmin = g_lossf(geom, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L)
        geom[i] += 1e-8
        df[i]  = (fplus - fmin)/(2e-8)
        
    return np.array(df)



def geometry_optimization(Urf, width, length, x_start, y_start, Z, M, dots, axis, omegas, learning_rate, stoch = 0, eps = 1e-8, L = 1e-6, step = 100):
    """
    Function, optimizating the shape of RF electrode from the individual_wells()
    function, to obtain individual wells with the required secular frequencies.
    Optmization is different from the reverse geometry optimization, since
    it optimizes the polygonal geometry for the arbitrary secular frequency set.

    Parameters
    ----------
    Urf: float
        Parametrized pseudopotential voltage
    width: float
        width of the RF electrode (vertical)
    length: float
        length of the RF electrode (horizontal)
    x_start: list shape (number of dots)
        starting array of (horizontal) widthes of the notches
        Starting geometry parameters must be sufficient to produce individual
        potential wells, or the optmization will fail.
    y_start: list shape (number of dots)
        starting array of (vertical) legthes of the notches
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
    x: list shape (number of dots)
        final array of optimized (horizontal) widthes of the notches
    y: list shape (number of dots)
        final array of optimized (vertical) legthes of the notches

    """
    
    geom = np.append(x_start, y_start)
    n_dots = np.array(dots).shape[0]
    
    loss2 = g_lossf(geom, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L)
    print("Initial loss:", loss2)
    
    m = g_precise_grad(geom, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L)
    v = np.square(m)
    b1 = 0.9
    b2 = 0.999
    big = np.arange(100000)
    small = np.arange(step)
    if stoch == 0:
        for t in big:
            for kk in small:
                grad = g_precise_grad(geom, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L)
                m = b1*m+(1-b1)*grad
                v = b2*v+(1-b2)*np.square(grad)
                mtilde = m/(1-b1**(1+t))
                vtilde = v/(1-b2**(1+t))
                v_sqrt = np.sqrt(vtilde) + np.ones(n_dots)*1e-8
                geom = geom - learning_rate*np.divide(mtilde,v_sqrt)
            loss2 = g_lossf(geom, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L)
            x = geom[0:n_dots]
            y = geom[n_dots:2*n_dots]
            print("Iteration:", t+1, "Loss function:", loss2)
            print("x =", list(x),"\ny =", list(y))
            if loss2 <= eps:
                break
    else:
        for t in big:
            for kk in small:
                grad = g_stoch_grad(geom, stoch, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L)
                m = b1*m+(1-b1)*grad
                v = b2*v+(1-b2)*np.square(grad)
                mtilde = m/(1-b1**(1+t))
                vtilde = v/(1-b2**(1+t))
                v_sqrt = np.sqrt(vtilde) + np.ones(n_dots)*1e-8
                geom = geom - learning_rate*np.divide(mtilde,v_sqrt)
            loss2 = g_lossf(geom, n_dots, dots, Urf, width, length, axis, omegas, Z, M, L)
            print("Iteration:", t+1, "Loss function:", loss2)
            x = geom[0:n_dots]
            y = geom[n_dots:2*n_dots]
            print("x =", list(x),"\ny =", list(y))
            if loss2 <= eps:
                break
            
    return x, y

