"""
Минимальный пример симуляции шаттлинга ионов с MPI-параллелизацией.
Использует SION для оптимизации напряжений и pylion/LAMMPS для молекулярной динамики.
"""
from __future__ import division
import os
import numpy as np
import scipy.constants as ct
from pathlib import Path
from sion.electrode import System, PolygonPixelElectrode
import sion as sn
import sion.pylion as pl  # requires SION installed: pip install -e . from repo

# ============================================
# Вспомогательные функции
# ============================================

def polygons_reshape(full_electrode_list, order, L=1e-6):
    """Переупорядочивает электроды согласно заданному порядку."""
    electrodes = []
    full_elec = []
    
    for i, elec in enumerate(order):
        electrodes.append([f'[{i}]', [full_electrode_list[elec] / L]])
        full_elec.append(np.array(full_electrode_list[elec]))
    
    s = System([
        PolygonPixelElectrode(cover_height=0, cover_nmax=0, name=n, paths=map(np.array, p))
        for n, p in electrodes
    ])
    
    return s, full_elec

# ============================================
# Параметры ловушки
# ============================================
L = 1e-6                        # Масштаб длины (мкм)
Vrf = 30                        # Пиковое RF напряжение (В)
M_Ca = 40 * ct.atomic_mass      # Масса иона Ca-40
Z = ct.elementary_charge        # Заряд иона
Omega = 2 * np.pi * 22e6        # RF частота (рад/с)
Urf = Vrf * np.sqrt(Z / M_Ca) / (2 * L * Omega)

# ============================================
# Загрузка и настройка электродов
# ============================================
_here = Path(__file__).resolve().parent
gds_path = _here / 'SQUAD_v4_simulate_transposed.GDS'
s, full_electrode_list = sn.polygons_from_gds(
    str(gds_path),
    need_plot=False,
    need_coordinates=True,
    L=L
)

# Порядок электродов
ORDER = [
    21, 1, 3, 5, 7, 9, 11, 12, 13, 14, 15, 16,
    2, 4, 6, 8, 10, 17, 18, 19, 22, 20, 23, 24, 0
]
s, full_elec = polygons_reshape(full_electrode_list, ORDER, L=L)

# RF напряжения
s.rfs = np.append([Urf], np.zeros(24))

# ============================================
# Оптимизация DC напряжений
# ============================================
s.dcs = np.zeros(25)
x0 = s.minimum([0, 0, 52.7], axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")

# Начальные DC напряжения
dc_set = [
    3., -3., 3., 0., 0., 0., 0., 0., 0., 0., 0.,
    3., -3., 3., 0., 0., 0., 0., 0., 0., 0., 0.,
    0.25, 0.18
]

# Оптимизация положения
dc_set = sn.position_optimization(
    s, [x0], dc_set, 
    numbers=[0, 1, 2, 11, 12, 13, 22, 23],
    eps=1e-2, tol=1e-18, 
    ion_masses=M_Ca, charges=1,
    callback_num=100, voltage_bounds=(-5, 5)
)
s.dcs = np.append(np.zeros(1), dc_set)

with s.with_voltages(dcs=s.dcs, rfs=None):
    x1 = s.minimum(x0 * 1.0001, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")

# ============================================
# Параметры шаттлинга
# ============================================
SHUTTLING_DISTANCE = 0.1    # Расстояние (мкм)
SHUTTLING_TIME = 1e-6       # Время (с)
V_MIN, V_MAX = -15, 15      # Границы напряжений (В)
RESOLUTION = 10             # Временное разрешение

x_start = [-1.60790355e-04, -5.90255245e-05, 5.96591864e+01]
shuttlers = np.array([1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 23, 24]) - 1

voltage_seq, funcs = sn.linear_shuttling_voltage(
    s, x_start, SHUTTLING_DISTANCE, SHUTTLING_TIME, dc_set,
    shuttlers=shuttlers, vmin=V_MIN, vmax=V_MAX, 
    res=RESOLUTION, need_func=True,
    freq_coeff=10, freq_ax=[1, 2]
)

# ============================================
# Создание симуляции
# ============================================
# Все файлы симуляции (.lammps, .lmp.log, .h5, dump) пишем в examples/simulation_files
SIMULATION_OUT_DIR = _here / 'simulation_files'
SIMULATION_OUT_DIR.mkdir(parents=True, exist_ok=True)
_orig_cwd = os.getcwd()
os.chdir(SIMULATION_OUT_DIR)

ion_number = 1
x_start_sim = np.array(x_start) * 1e-6
positions = sn.ioncloud_min(x_start_sim, ion_number, 5e-6)

sim = pl.Simulation(Path(__file__).stem)

Ca_ions = {'mass': 40, 'charge': 1}
RF_electrodes = [full_elec[0]]
DC_electrodes = [full_elec[i] for i in range(1, 25)]

sim.append(pl.placeions(Ca_ions, positions))
sim.append(sn.polygon_shuttling([Omega], [Vrf], RF_electrodes, DC_electrodes, funcs))
sim.append(pl.langevinbath(0, 5e-6))
sim.append(pl.dump('poslinshuttle.txt', variables=['x', 'y', 'z'], steps=10))

t_evolve = int(SHUTTLING_TIME * 20 * Omega)
print(f'Evolution: {t_evolve} steps')

sim.append(pl.evolve(t_evolve))

# Параллелизация: mpi_processes × omp_threads (SION's bundled pylion)
sim.execute(mpi_processes=4, omp_threads=1)

os.chdir(_orig_cwd)  # вернуться в исходную директорию
