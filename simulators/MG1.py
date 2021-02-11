import torch
import numpy as np

class MG1():
    def __init__(self, args):
        self.args = args

    def run(self, thetas, observation):
        if observation:
            thetas = thetas.repeat(100, 1)

        y_t = torch.Tensor([0.] * thetas.shape[0]).reshape(-1, thetas.shape[0]).to(self.args.device)
        thetas = torch.t(thetas).to(self.args.device)

        a = torch.zeros(1, thetas.shape[1]).to(self.args.device)
        d = torch.zeros(1, thetas.shape[1]).to(self.args.device)
        zero = torch.zeros(1, thetas.shape[1]).to(self.args.device)

        for t in range(self.args.numTime):
            s = thetas[0] + thetas[1] * torch.FloatTensor(1, thetas.shape[1]).uniform_(0, 1).to(self.args.device)
            a = a + torch.distributions.exponential.Exponential(thetas[2]).rsample().reshape(1, -1).to(self.args.device)
            obs = s + torch.max(zero, a - d)
            d = d + obs
            y_t = torch.cat((y_t, obs))

        y_t = y_t[1:]

        mean = torch.t(torch.Tensor(
                np.quantile(y_t.cpu().detach().numpy(), (1. / (self.args.xDim - 1)) * np.arange(self.args.xDim),
                            axis=0)).to(self.args.device)).detach()

        if observation:
            return torch.mean(mean, 0).reshape(1, -1)

        else:
            return mean