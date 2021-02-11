import torch
from sampler.Metropolis_Hastings_Sampler import Metropolis_Hastings_Sampler

def sample_from_likelihood(self, num_samples, init=False):
    # On initial round, select random batch of simulation inputs
    # On consecutive rounds, select batch of inputs from the inferred posterior using the Bayesian rule on the neural likelihood
    if init:
        return torch.Tensor(self.prior.gen(num_samples)).to(self.args.device).detach()
    else:
        return Metropolis_Hastings_Sampler(self, self.netLikelihood, num_samples, num_chains=num_samples).detach()