import torch
from sampler.Metropolis_Hastings_Sampler import Metropolis_Hastings_Sampler

def sample_from_exact_likelihood(self, num_samples):
    return Metropolis_Hastings_Sampler(self, self.sim.simulator, num_samples, num_chains=num_samples).detach()