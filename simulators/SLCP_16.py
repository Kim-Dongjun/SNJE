import torch

class SLCP_16():
    def __init__(self, args):
        self.args = args

    def run(self, thetas, observation):
        mean = thetas[:,:2] ** 2

        diag1 = thetas[:,2].reshape(-1,1) ** 2
        diag2 = thetas[:,3].reshape(-1,1) ** 2
        corr = torch.tanh(thetas[:,4]).reshape(-1,1)
        cov = torch.cat((diag1 ** 2, corr * diag1 * diag2, corr * diag1 * diag2, diag2 ** 2), 1).reshape(-1,2,2)

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov + 1e-6 * torch.eye(2).to(self.args.device))

        return distribution.sample([int(self.args.xDim/mean.shape[1])]).transpose(1,0).reshape(-1, self.args.xDim).detach()

    def log_prob(self, context='', inputs=''):
        mean = context[:, :2] ** 2

        diag1 = context[:, 2].reshape(-1, 1) ** 2
        diag2 = context[:, 3].reshape(-1, 1) ** 2
        corr = torch.tanh(context[:, 4]).reshape(-1, 1)
        cov = torch.cat((diag1 ** 2, corr * diag1 * diag2, corr * diag1 * diag2, diag2 ** 2), 1).reshape(-1, 2, 2)

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov + 1e-6 * torch.eye(2).to(self.args.device))
        ll = torch.zeros((context.shape[0])).to(self.args.device)
        for k in range(int(inputs.shape[1] / 2)):
            ll = ll + distribution.log_prob(inputs[:,2 * k: 2 * (k + 1)])

        return ll.detach()