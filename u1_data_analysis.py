import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from test import *
import glob
import json

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "-input", dest="input_dir", help="directory from which to get data files for analysis")
args = argParser.parse_args()

directory = args.input_dir

#Get simulation run parameters
with open(directory + "/parameters.json", 'r') as f:
    args = json.load(f)

beta = args['beta']
#Save configurations in this dictionary
configs = {}

#Load configurations from directory files
for i, file in enumerate(natsorted(glob.glob(directory + "/configs_*.npz"), key=lambda x: x.lower())):
    with np.load(file) as data:
        configs[i] = data['configs']

#Analysis portion

#Measure observables

observables = {
    'Topological charge': topological_charge,
    'Topological susc.': topological_susc,
    'Avg. Plaquette': compute_avg_plaq,
    'Polyakov loop': abs_polyakov,
    'Polyakov susc.': polyakov_susc,
    '4x4 Wilson loop': wilson
}

obs = {k:[] for k in observables}

for i, cfgs in configs.items():
    for name, func in observables.items():
        obs[name].append(jackknife(func(cfgs)))

for k in obs:
    obs[k] = np.asarray((obs[k]))


N_col = 3
N_rows = int(len(observables)/3)

fig, axes = plt.subplots(N_rows, N_col, dpi=75, figsize=(12, 6))
fig.suptitle(f"$U(1)$ gauge Langevin simulations || L = {args['lattice_size']}, $\\kappa \in [{beta[0]}, {beta[-1]}]$")

for ax, (name, values) in zip(np.ravel(axes), obs.items()):
    y, y_err = values[:, 0], values[:, 1]
    ax.errorbar(beta, y, y_err, capsize=2.0, ecolor='black', label=name)
    ax.legend(prop={'size':6})
    ax.set_xlim(beta[0], beta[-1])
    ax.grid()

plt.tight_layout()
plt.show()