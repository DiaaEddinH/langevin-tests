from test import LangevinSystem, ScalarPhi4Action, rng
import os
import argparse
import json

argParser = argparse.ArgumentParser()
argParser.add_argument("-o", "-output", dest="output_file", help="file/directory in which to save data files")

argParser.add_argument("-L", "--lattice", dest = "lattice_size", type=int, help="Length of a square lattice")
argParser.add_argument("-k", "--kappa", type=float, dest='kappa')
argParser.add_argument("-l", "--lmbda", type=float, dest='lmbda', default=0.022)
argParser.add_argument("--step_size", type=float, dest='step_size', default=0.1)

argParser.add_argument("--therm_steps", type=int, dest='therm_steps', help="Number of thermalisation steps")
argParser.add_argument("--gen_steps", type=int, dest='gen_steps', help="Number of configuration generation steps")
argParser.add_argument("--save_interval", type=int, dest='save_interval', default=10, help="Interval between each configuration saved")

args = argParser.parse_args()

os.makedirs(args.output_file, exist_ok=True)
with open(args.output_file + '/arg_parameters.json', 'w+') as f:
    json.dump(vars(args), f, indent=4)


L = args.lattice_size
lattice_shape = (L, L)
init_configs = rng.standard_normal(lattice_shape)

#PHI4 PARAMETERS
kappa = args.kappa; lmbda = args.lmbda

action = ScalarPhi4Action(kappa = kappa, lmbda = lmbda)

print(5*'-' + f"Running for L={L}, kappa={kappa}, lambda={lmbda}" + 5*'-')
#Langevin parameters
step_size = args.step_size; N_therm = args.therm_steps; N_gap=args.save_interval; N_gen = args.gen_steps

filepath =  args.output_file + "/configs_0"


LS = LangevinSystem(
    init_configs=init_configs, drift=action.drift
)

LS.run(
    therm_steps=N_therm, 
    generate_steps=N_gen, 
    step_size=step_size, 
    save_interval=N_gap,
    output_file=filepath
    )

# observables = {
#     'magnetisation': magnetisation,
#     'abs. magnetisation': abs_magnetisation,
#     'magn. susc.': magnetic_susc,
#     'energy': action.energy,
#     'heat capacity': action.heat_capacity,
#     'Binder cumulant': binder_cumulant
# }

# for name, obs in observables.items():
#     avg, err = jackknife(obs(configs))
#     print(f"{name} = {avg: .5f} +/- {err: .5f}")

# mag = magnetisation(configs)
# ac, ac_err = autocorrelations(mag, time_window=100)

# fig, ax = plt.subplots(1, 1, dpi=75, figsize=(12,6))

# ax.errorbar(range(0, len(ac)), y=ac, yerr=ac_err, ecolor='black', capsize=2.0)
# ax.set_xlabel("MC time")
# ax.set_ylabel("Autocorrelation")
# #plt.yscale('log')
# plt.grid()
# plt.show()

# fig, axes = plt.subplots(2, 3, dpi = 125, figsize=(12, 6))
# for ax in axes.ravel():
#     cfg_f = rng.choice(configs)
#     ax.imshow(cfg_f)

#     #axes[1, i].imshow(rng.standard_normal(cfg_f.shape))
# plt.show()

# fig, ax = plt.subplots(1, 2, dpi = 75, figsize=(12,6))

# #Magnetisation distribution
# init_hist = ax[0].hist(magnetisation(rng.standard_normal(configs.shape)), bins=50, histtype='step', density=True, label='Initial distr.')
# final_hist = ax[0].hist(magnetisation(configs), bins=50, histtype='step', density=True, label='Equilibrium distr.')
# ax[0].set_title("Magnetisation distribution")

# #Configuration distribution
# init_hist = ax[1].hist(rng.standard_normal(configs.shape).flatten(), bins=100, histtype='step', density=True, label='Initial distr.')
# final_hist = ax[1].hist(configs.flatten(), bins=100, histtype='step', density=True, label='Equilibrium distr.')
# ax[1].set_title("Configuration distribution")

# plt.legend()
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(1, 1, dpi = 75, figsize=(12,6))

# c = two_point_corr(configs)
# y, y_err = c[:,0], c[:,1]
# x = np.array(range(1, len(y)+1))
# ax.errorbar(x, y, y_err, ecolor='black', capsize=2.0)
# ax.set_xlabel("Lattice point")
# ax.set_title("2-pt correlation function")

# plt.tight_layout()
# plt.show()