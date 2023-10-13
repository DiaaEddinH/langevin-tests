from test import *
import argparse
import json
import os
from tqdm import tqdm

argParser = argparse.ArgumentParser()
argParser.add_argument("-o", "-output", dest="output_file", help="file/directory in which to save data files")

argParser.add_argument("-L", "--lattice", dest = "lattice_size", type=int, help="Length of a square lattice")
argParser.add_argument("-min_k", "--kappa_min", type=float, dest='kappa_min')
argParser.add_argument("-max_k", "--kappa_max", type=float, dest='kappa_max')
argParser.add_argument("-n", type=int, dest="N", default=10)
argParser.add_argument("-l", "--lmbda", type=float, dest='lmbda', default=0.022)

argParser.add_argument("--step_size", type=float, dest='step_size', default=0.1, help="Step size for Langevin evolution")
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
lmbda = args.lmbda

#Langevin parameters
step_size = args.step_size; N_therm = args.therm_steps; N_gap=args.save_interval; N_gen = args.gen_steps

kappa_range = np.linspace(args.kappa_min, args.kappa_max, args.N)

for i, kappa in enumerate(tqdm(kappa_range)):
    filepath =  args.output_file + f"/configs_{i}"

    action = ScalarPhi4Action(kappa = kappa, lmbda = lmbda)

    LS = LangevinSystem(
        init_configs=init_configs, drift=action.drift
    )

    LS.run(
        therm_steps=N_therm, 
        generate_steps=N_gen, 
        step_size=step_size, 
        save_interval=N_gap,
        output_file= filepath
        )