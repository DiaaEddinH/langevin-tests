from test import *
import numpy as np

import argparse
import json

argParser = argparse.ArgumentParser()
argParser.add_argument("-o", "-output", dest="output_file", default = None, help="file/directory in which to save data files")

argParser.add_argument("-L", "--lattice", dest = "lattice_size", type=int, help="Length of a square lattice")
argParser.add_argument("-b", "--beta", type=float, dest='beta')
argParser.add_argument("--step_size", type=float, dest='step_size', default=0.1)

argParser.add_argument("--therm_steps", type=int, dest='therm_steps', help="Number of thermalisation steps")
argParser.add_argument("--gen_steps", type=int, dest='gen_steps', help="Number of configuration generation steps")
argParser.add_argument("--save_interval", type=int, dest='save_interval', default=10, help="Interval between each configuration saved")

args = argParser.parse_args()


L = args.lattice_size
lattice_shape = (L, L)
link_shape = (2, *lattice_shape)

init_configs = rng.uniform(0, 1, link_shape)

#U(1) PARAMETERS
action = U1GaugeAction(beta=args.beta)

print(5*'-' + f"Running for L={L}, beta = {args.beta}" + 5*'-')
#Langevin parameters
step_size = args.step_size; N_therm = args.therm_steps; N_gap=args.save_interval; N_gen = args.gen_steps

LS = LangevinSystem(
    init_configs=init_configs, drift=action.drift
)

LS.run(
    therm_steps=N_therm, 
    generate_steps=N_gen, 
    step_size=step_size, 
    save_interval=N_gap,
    output_file=args.output_file
    )

configs = np.array(LS.configs)

observables = {
    'Topological charge': topological_charge,
    'Topological susc.': topological_susc,
    # 'Avg. Plaquette': compute_avg_plaq,
    # 'Polyakov loop': abs_polyakov,
    # 'Polyakov susc.': polyakov_susc,
    # '4x4 Wilson loop': wilson
}

obs_ac = {}
for name, obs in observables.items():
  obs_ac[name] = autocorrelations(obs(configs), time_window = 100)

fig, ax = plt.subplots(1, 1, dpi=125, figsize=(12,6))

for name, (ac, ac_err) in obs_ac.items():
  ax.errorbar(range(0, len(ac)), y=ac, yerr=ac_err, ecolor='black', capsize=2.0, label=name)

ax.set_xlabel("MC time")
ax.set_ylabel("Autocorrelation")
#plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(1, 1, dpi=125, figsize=(12,6))

ax.plot(topological_charge(configs))
ax.set_xlabel("MC time")
ax.set_ylabel("Autocorrelation")
#plt.yscale('log')
plt.grid()
plt.show()


# cP = polyakov(configs)
# P = abs_polyakov(configs)

# fig, ax = plt.subplots(1, 1, dpi=75, figsize=(6,6))
# ax.scatter(cP.real, cP.imag)
# ax.set_xlabel("ReP")
# ax.set_xlabel("ImP")
# plt.show()

# ac, ac_err = autocorrelations(P, time_window=100)

# fig, ax = plt.subplots(1, 1, dpi=75, figsize=(12,6))

# ax.errorbar(range(0, len(ac)), y=ac, yerr=ac_err, ecolor='black', capsize=2.0)
# ax.set_xlabel("MC time")
# ax.set_ylabel("Autocorrelation")
# #plt.yscale('log')
# plt.grid()
# plt.show()

# configs = np.asarray(LS.configs)

# topo_charge = topological_charge(configs)
# np.testing.assert_allclose(topo_charge, np.around(topo_charge), atol=1e-6, err_msg="Topological charge is an integer quantity")

# observables = {
#     'Topological charge': topological_charge,
#     'Topological susc.': topological_susc,
#     'Avg. Plaquette': compute_avg_plaq,
#     'Polyakov loop': abs_polyakov,
#     'Polyakov susc.': polyakov_susc,
#     '2x2 Wilson loop': wilson
# }

# for name, obs in observables.items():
#     avg, err = jackknife(obs(configs))
#     print(f"{name} = {avg: .5f} +/- {err: .5f}")

# fig, ax = plt.subplots(1, 2, dpi = 75, figsize=(12,6))
# ax[0].plot(topological_charge(configs))
# ax[0].set_ylabel('Top. Charge')
# ax[1].plot(topological_susc(configs))
# ax[1].set_ylabel('Top. Susc.')

# fig, ax = plt.subplots(1, 2, dpi = 75, figsize=(12,6))

# #Topological distribution
# init_hist = ax[0].hist(compute_avg_plaq(rng.standard_normal(configs.shape)), bins=50, histtype='step', density=True, label='Initial distr.')
# final_hist = ax[0].hist(compute_avg_plaq(configs), bins=50, histtype='step', density=True, label='Equilibrium distr.')
# ax[0].set_title("Avg Plaquette distribution")

# #Configuration distribution
# init_hist = ax[1].hist(abs_polyakov(rng.standard_normal(configs.shape)), bins=100, histtype='step', density=True, label='Initial distr.')
# final_hist = ax[1].hist(abs_polyakov(configs), bins=100, histtype='step', density=True, label='Equilibrium distr.')
# ax[1].set_title("Abs Polyakov Loop distribution")

# plt.legend()
# plt.tight_layout()
# plt.show()