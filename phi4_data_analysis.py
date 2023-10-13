import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from test import *
import glob
import json

#Get simulation run parameters
with open("range_run/arg_parameters.json", 'r') as f:
    args = json.load(f)

#Save configurations in this dictionary
configs = {}

#Load configurations from directory files
for i, file in enumerate(natsorted(glob.glob("range_run/configs_*.npz"), key=lambda x: x.lower())):
    with np.load(file) as data:
        configs[i] = data['configs']

#Analysis portion

kappa = np.linspace(args['kappa_min'], args['kappa_max'], args['N'])

#Measure observables
measurables = ['magnetisation', 'abs. magnetisation', 'magn. susc.', 'energy', 'heat capacity', 'Binder cumulant']
obs = {k:[] for k in measurables}

for i, cfgs in configs.items():
    action = ScalarPhi4Action(kappa=kappa[i], lmbda=args['lmbda'])

    observables = {
        'magnetisation': magnetisation,
        'abs. magnetisation': abs_magnetisation,
        'magn. susc.': magnetic_susc,
        'energy': action.energy,
        'heat capacity': action.heat_capacity,
        'Binder cumulant': binder_cumulant
    }

    for name, func in observables.items():
        obs[name].append(jackknife(func(cfgs)))

for k in obs:
    obs[k] = np.asarray((obs[k]))


N_col = 3
N_rows = int(len(observables)/3)

fig, axes = plt.subplots(N_rows, N_col, dpi=75, figsize=(12, 6))
fig.suptitle(f"$\\phi^4$ Langevin simulations || L = {args['lattice_size']}, $\\lambda$ = {args['lmbda']}, $\\kappa \in [{kappa[0]}, {kappa[-1]}]$")

for ax, (name, values) in zip(np.ravel(axes), obs.items()):
    y, y_err = values[:, 0], values[:, 1]
    ax.errorbar(kappa, y, y_err, capsize=2.0, ecolor='black', label=name)
    ax.legend(prop={'size':6})
    ax.set_xlim(kappa[0], kappa[-1])
    ax.grid()

plt.tight_layout()
plt.show()
