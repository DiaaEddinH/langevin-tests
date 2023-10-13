from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm

rng = np.random.default_rng(2105232984)

def autocorrelations(observable: np.ndarray, time_window: int) -> np.ndarray:
    #obs: shape (N_config, )
    #lagged_obs = (time_window, N_config)
    lagged_obs = np.array([observable * np.roll(observable, -t, 0) for t in range(time_window)])
    obs_sq = np.mean(observable)**2

    J, J_err = jackknife(lagged_obs.T - obs_sq)

    return J/J[0], J_err/J[0]


def bootstrap(x: np.ndarray, Nboot: int, binsize: int) -> tuple:
    global rng
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        x_bin = x[rng.integers(len(x), size=len(x))]
        boots.append(np.mean(x_bin, axis=(0,1)))
    return np.mean(boots), np.std(boots)

def jackknife(samples: np.ndarray):
    """Return mean and estimated lower error bound."""
    means = []

    for i in range(samples.shape[0]):
        means.append(np.delete(samples, i, axis=0).mean(axis=0))

    means = np.asarray(means)
    mean = means.mean(axis=0)
    error = np.sqrt((samples.shape[0] - 1) * np.mean(np.square(means - mean), axis=0))
    
    return mean, error

def compute_plaq(links, mu, nu):
  return (links[mu] + np.roll(links[nu], -1, mu) - np.roll(links[mu], -1, nu) - links[nu])
   

class U1GaugeAction:
  def __init__(self, beta: float) -> None:
    self.beta = beta

  def __call__(self, cfgs: np.ndarray) -> float:
    return np.sum(self.density(cfgs))

  def energy(self, cfgs: np.ndarray) -> np.ndarray:
    dims = tuple(range(1, len(cfgs.shape)))
    return np.mean([self.density(cfg) for cfg in cfgs], axis=dims)

  def heat_capacity(self, cfgs: np.ndarray) -> np.ndarray:
    vol = np.prod(cfgs.shape[1:])
    energy = self.energy(cfgs)
    return vol * (energy - np.mean(energy))**2

  def density(self, cfgs: np.ndarray) -> np.ndarray:
    Nd = cfgs.shape[0]
    action_density = 0
    for mu in range(Nd):
      for nu in range(mu+1):
        action_density += 1 - np.cos(compute_plaq(cfgs, mu, nu))
    return self.beta * action_density

  def drift(self, cfgs: np.ndarray) -> np.ndarray:
    Nd = cfgs.shape[0]
    action_grad = []
    for mu in range(Nd):
      grad_ = 0
      for nu in [i for i in range(Nd) if i != mu]:
        plaq = compute_plaq(cfgs, mu, nu)
        grad_ += np.sin(plaq) - np.sin(np.roll(plaq, 1, nu))
      action_grad.append(grad_)
    return -self.beta * np.array(action_grad)

class ScalarPhi4Action:
    def __init__(self, kappa: float, lmbda: float) -> None:
        self.kappa = kappa
        self.lmbda = lmbda    
    
    def __call__(self, cfgs: np.ndarray) -> float:
       return np.sum(self.density(cfgs))

    def energy(self, cfgs: np.ndarray) -> np.ndarray:
        dims = tuple(range(1, len(cfgs.shape)))
        return np.mean([self.density(cfg) for cfg in cfgs], axis=dims)
    
    def heat_capacity(self, cfgs: np.ndarray) -> np.ndarray:
        vol = np.prod(cfgs.shape[1:])
        energy = self.energy(cfgs)
        return vol * (energy - energy.mean())**2
    
    def density(self, cfgs: np.ndarray) -> np.ndarray:
        Nd = len(cfgs.shape)
        action_density = (1 - 2*self.lmbda) * cfgs**2 + self.lmbda * cfgs**4
        for mu in range(Nd):
            action_density += -2 * self.kappa * cfgs * np.roll(cfgs, -1, mu)
        return action_density
    
    def drift(self, cfgs: np.ndarray) -> np.ndarray:
        Nd = len(cfgs.shape)
        action_density = 2*cfgs * (1 + 2 * self.lmbda * (cfgs**2 - 1))
        for mu in range(Nd):
            action_density += -2 * self.kappa * (np.roll(cfgs, -1, mu) + np.roll(cfgs, 1, mu))
        return -action_density

class U1GaugeLangevinSystem:
    def __init__(self, init_configs: np.ndarray, drift: np.ndarray) -> None:
        self.drift = drift
        self.current_configs = init_configs
        self.configs = []
    
    def save_configs(self, output_file: str):
        np.savez(file=output_file, configs = np.asarray(self.configs))
    
    def evolve(self, time_steps: int, step_size: float, save_configs: bool=False, save_interval: int = 1) -> None:
        for i in range(time_steps):
            old_configs = self.current_configs
            noise = rng.normal(loc=0., scale=np.sqrt(step_size), size=old_configs.shape)
            self.current_configs = old_configs + self.drift(old_configs) * step_size + noise
            self.current_configs = project_to_u1(self.current_configs)
        
            if save_configs:
                if i % save_interval == 0:
                    self.configs.append(self.current_configs)
    
    def run(self, therm_steps: int, generate_steps: int, step_size: float, save_interval: int, output_file: Optional[str] = None) -> None:
        #Thermalization step
        self.evolve(time_steps=therm_steps, step_size=step_size)
        
        #Configuration generation step
        self.evolve(generate_steps, step_size, save_configs=True, save_interval=save_interval)

        #Save configs to file if 'output_file' is not None
        if output_file:
            self.save_configs(output_file=output_file)

class LangevinSystem:
    def __init__(self, init_configs: np.ndarray, drift: np.ndarray) -> None:
        self.drift = drift
        self.drift_tuning_factor = 1
        self.current_configs = init_configs
        self.configs = []
    
    def save_configs(self, output_file: str):
        np.savez(file=output_file, configs = np.asarray(self.configs))
    
    def adapt_step_size(self, step_size: float, tuning_factor: float = 2, tol: float=2e-4) -> float:
       current_drift = self.drift(self.current_configs)
       max_drift = np.max(np.abs(current_drift))
       proposed_step_size = step_size * self.drift_tuning_factor / max_drift
       eK = step_size * max_drift
       lower_bound = tol / tuning_factor; upper_bound = tol * tuning_factor

       if lower_bound <= eK <= upper_bound:
          return step_size
       else:
          if eK >= upper_bound:
             return step_size / tuning_factor
          else:
             return step_size * tuning_factor
             
    
    def evolve(self, time_steps: int, step_size: float, save_configs: bool=False, save_interval: int = 1) -> None:
        step_evolve = []
        for i in range(time_steps):
            old_configs = self.current_configs
            noise = rng.standard_normal(size=old_configs.shape)
            self.current_configs = old_configs + self.drift(old_configs) * step_size + np.sqrt(step_size) * noise
        
            if save_configs:
                if i % save_interval == 0:
                    self.configs.append(self.current_configs)

        #     step_size = self.adapt_step_size(step_size)
        #     step_evolve.append(step_size)

        # fig, ax = plt.subplots(1, 1, dpi=75, figsize=(12, 6))
        # ax.plot(step_evolve)
        # ax.set_ylabel("Step size")
        # ax.set_xlabel("Langevin time")

        # plt.show()
    
    def run(self, therm_steps: int, generate_steps: int, step_size: float, save_interval: int, output_file: Optional[str] = None) -> None:
        #Thermalization step
        self.evolve(time_steps=therm_steps, step_size=step_size)
        
        #Configuration generation step
        self.evolve(generate_steps, step_size, save_configs=True, save_interval=save_interval)

        #Save configs to file if 'output_file' is not None
        if output_file:
            self.save_configs(output_file=output_file)

def extract_observables(history: dict, observables: dict) -> dict:
    out = {k: [] for k in observables}

    for kappa, cfgs in history.items():
        for name, obs in observables.items():
            out[name].append(jackknife(obs(cfgs)))
        
    for k in out:
        out[k] = np.asarray(out[k])
    
    return out

def plot_observables(kappa_range: list, observables: dict, *args, **kwargs) -> None:
    N_col = 3
    N_rows = int(len(observables)/3)

    fig, axes = plt.subplots(N_rows, N_col, *args, **kwargs)

    for ax, (name, values) in zip(np.ravel(axes), observables.items()):
        y, y_err = values[:, 0], values[:, 1]
        ax.errorbar(kappa_range, y, y_err, capsize=2.0, ecolor='black', label=name)
        ax.legend(prop={'size':6})
        ax.set_xlim(kappa_range[0], kappa_range[-1])
        ax.grid()
    
    plt.tight_layout()
    plt.show()


def magnetisation(cfgs: np.ndarray) -> np.ndarray:
    dims = tuple(range(1, len(cfgs.shape)))
    return np.mean(cfgs, axis=dims)

def abs_magnetisation(cfgs):
  return magnetisation(np.abs(cfgs))

def magnetic_susc(cfgs: np.ndarray) -> np.ndarray:
    vol = np.prod(cfgs.shape[1:])
    M = magnetisation(cfgs)
    return vol * (M - M.mean())**2

def binder_cumulant(cfgs: np.ndarray) -> np.ndarray:
  M = magnetisation(cfgs)
  return 1 - M**4/(3 * np.mean(M*M)**2)

def two_point_corr(cfgs: np.ndarray) -> np.ndarray:
  corr_func = []
  dims = tuple(range(1, len(cfgs.shape)))
  for i in range(1, cfgs.shape[1]):
    corrs = []

    for mu in range(len(cfgs.shape) - 1):
      corrs.append(np.mean(cfgs * np.roll(cfgs, -i, mu+1), axis = dims))

    corrs = np.mean(corrs, axis=0)
    corr_func.append(jackknife(corrs - np.mean(cfgs)**2))

  return np.array(corr_func)


def project_to_u1(cfgs: np.ndarray) -> np.ndarray: #Projecting to [-π, π]
  return np.remainder(cfgs + np.pi, 2*np.pi) - np.pi

def topological_charge(cfgs: np.ndarray) -> np.ndarray:
  Q = []
  for link in cfgs:
    P01 = project_to_u1(compute_plaq(link, 0, 1))
    Q.append(
        np.sum(P01)/(2*np.pi)
    )

  return np.asarray(Q)

def compute_avg_plaq(cfgs):
  return np.mean([np.cos(compute_plaq(link, 0, 1)) for link in cfgs], axis=(1,2))

def topological_susc(cfgs: np.ndarray) -> np.ndarray:
   return topological_charge(cfgs=cfgs)**2

def polyakov(cfgs):
  return np.exp(1j*np.sum(cfgs[:,0], axis = 1)).mean(axis=1)

def abs_polyakov(cfgs):
  return np.abs(polyakov(cfgs))

def polyakov_susc(cfgs):
  P = abs_polyakov(cfgs)
  return P**2 - np.mean(P)**2

def wilson(links: np.ndarray, T:int=4, R:int=4):
  W = 0
  for t in range(T):
    W += np.roll(links[:, 0], (-t, -R), (1, 2)) # (t, x+R) -> (t+T, x+R)
    W += -np.roll(links[:, 0], (-t, 0), (1, 2)) # (t, x) <- (t+T, x)
  for r in range(R):
    W += np.roll(links[:, 1], (0, -r), (1, 2)) # (t, x) -> (t, x+R)
    W += -np.roll(links[:, 1], (-T, -r), (1, 2)) # (t+T, x+R) <- (t+T, x)

  return np.cos(W).mean((1,2)) # 1/V Σ_i ( Re W_c[U_i])