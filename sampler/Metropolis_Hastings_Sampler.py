import torch
import math

def Metropolis_Hastings_Sampler(self, netLikelihood, n_sample, num_chains=None, thin=10, proposal_std=0.1):
    # Implementation of the M-H sampler
    # thinning method is adopted to reduce the auto-correlation of MCMC samples
    observation = self.observation
    if observation.shape[0] == self.args.xDim:
        observation = observation.repeat(num_chains, 1)
    elif observation.shape[0] != self.args.xDim:
        observation = observation.repeat(1, num_chains // observation.shape[0]).reshape(-1,self.args.xDim)

    thetas = torch.rand((num_chains, self.args.thetaDim)).to(self.args.device)
    mcmc_samples = torch.Tensor([]).to(self.args.device)
    proposal_std = torch.Tensor([[proposal_std]]).repeat(num_chains, self.args.thetaDim).to(self.args.device)
    for itr in range(self.args.burnInMCMC + thin * math.ceil(n_sample / num_chains)):
        try:
            thetas_intermediate = torch.clamp(thetas + proposal_std * torch.randn((num_chains, self.args.thetaDim)).to(self.args.device), min=0, max=1)
            rand = torch.rand(num_chains).to(self.args.device).reshape(-1)
            mask = (torch.exp(
                torch.min(netLikelihood.log_prob(x=observation, theta=self.sim.min + (self.sim.max - self.sim.min) * thetas_intermediate).reshape(-1)
                            + torch.Tensor(self.prior.eval(self.sim.min + (self.sim.max - self.sim.min) * thetas_intermediate).reshape(-1)).to(self.args.device)
                          - netLikelihood.log_prob(x=observation, theta=self.sim.min + (self.sim.max - self.sim.min) * thetas).reshape(-1)
                          - torch.Tensor(self.prior.eval(self.sim.min + (self.sim.max - self.sim.min) * thetas).reshape(-1)).to(self.args.device),
                          torch.Tensor([0.] * num_chains).to(self.args.device))) > rand).float().reshape(-1, 1)
        except:
            try:
                thetas_intermediate = thetas + proposal_std * torch.randn((num_chains, self.args.thetaDim)).to(self.args.device)
                rand = torch.rand(num_chains).to(self.args.device).reshape(-1)
                mask = (torch.exp(
                    torch.min(netLikelihood.log_prob(context=self.sim.min + (self.sim.max - self.sim.min) * thetas_intermediate, inputs=observation).reshape(-1)
                              + torch.Tensor(self.prior.eval(self.sim.min + (self.sim.max - self.sim.min) * thetas_intermediate).reshape(-1)).to(self.args.device)
                              - netLikelihood.log_prob(context=self.sim.min + (self.sim.max - self.sim.min) * thetas.to(self.args.device), inputs=observation).reshape(-1)
                              - torch.Tensor(self.prior.eval(self.sim.min + (self.sim.max - self.sim.min) * thetas).reshape(-1)).to(self.args.device),
                              torch.Tensor([0.] * num_chains).to(self.args.device))) > rand).float().reshape(-1, 1)
            except:
                thetas_intermediate = thetas + proposal_std * torch.randn((num_chains, self.args.thetaDim)).to(self.args.device)
                rand = torch.rand(num_chains).to(self.args.device).reshape(-1)
                mask = (torch.exp(
                    torch.min(netLikelihood(
                        context=self.sim.min + (self.sim.max - self.sim.min) * thetas_intermediate,
                        inputs=observation).reshape(-1)
                              + torch.Tensor(
                        self.prior.eval(self.sim.min + (self.sim.max - self.sim.min) * thetas_intermediate).reshape(
                            -1)).to(self.args.device)
                              - netLikelihood(
                        context=self.sim.min + (self.sim.max - self.sim.min) * thetas.to(self.args.device),
                        inputs=observation).reshape(-1)
                              - torch.Tensor(
                        self.prior.eval(self.sim.min + (self.sim.max - self.sim.min) * thetas).reshape(-1)).to(
                        self.args.device),
                              torch.Tensor([0.] * num_chains).to(self.args.device))) > rand).float().reshape(-1, 1)
        if itr == 0:
            masks = mask.reshape(1, -1, 1)
        else:
            masks = torch.cat((masks, mask.reshape(1, -1, 1)))
        if itr % thin == 0:
            # "optimal" acceptance ratio becomes 23.4% (https://projecteuclid.org/euclid.aoap/1034625254)
            bool = (torch.sum(masks[-100:,:,:],0) / 100 > 0.234).float()
            proposal_std = (1.1 * bool + 0.9 * (1 - bool)).repeat(1,self.args.thetaDim).to(self.args.device) * proposal_std
        thetas = thetas_intermediate * mask + thetas.to(self.args.device) * (1 - mask)
        if max(itr - self.args.burnInMCMC, 0) % thin == thin - 1:
            mcmc_samples = torch.cat((mcmc_samples, thetas))
    return self.sim.min + (self.sim.max - self.sim.min) * mcmc_samples[:n_sample]