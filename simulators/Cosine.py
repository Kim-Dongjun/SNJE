import torch
import numpy as np

class Cosine():
    def __init__(self, args):
        self.args = args

    def run(self, thetas, observation):
        mean = torch.cos(thetas * 5 * np.pi)

        cov = 0.1 * torch.eye((2)).repeat(thetas.shape[0], 1).reshape(thetas.shape[0], 2, 2).to(self.args.device)

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)

        if observation:
            return torch.Tensor([0.0] * self.args.xDim).to(self.args.device).detach().reshape(1, -1)
        else:
            return distribution.sample([int(self.args.xDim / self.args.thetaDim)]).transpose(1, 0).reshape(thetas.shape[0], -1)

    def log_prob(self, context='', inputs=''):
        mean = torch.cos(context * 5 * np.pi)

        cov = 0.1 * torch.eye((2)).repeat(context.shape[0], 1).reshape(context.shape[0], 2, 2).to(self.args.device)

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)

        ll = torch.zeros((context.shape[0])).to(self.args.device)
        for k in range(int(inputs.shape[1] / context.shape[1])):
            ll = ll + distribution.log_prob(inputs[:, context.shape[1] * k: context.shape[1] * (k + 1)])

        return ll.detach()