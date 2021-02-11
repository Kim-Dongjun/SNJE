import torch

class SLCP_256():
    def __init__(self, args):
        self.args = args

    def run(self, thetas, observation):

        mean = thetas ** 2

        cov = torch.eye(thetas.shape[1]).to(self.args.device)

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)

        return distribution.sample([int(self.args.xDim / mean.shape[1])]).transpose(1, 0).reshape(-1, self.args.xDim).detach()

    def log_prob(self, context='', inputs=''):

        mean = context ** 2

        cov = torch.eye(context.shape[1]).to(self.args.device)

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
        ll = torch.zeros((context.shape[0])).to(self.args.device)
        # print("!!! : ", x.shape, thetas.shape)
        for k in range(int(inputs.shape[1] / 8)):
            ll = ll + distribution.log_prob(inputs[:, 8 * k: 8 * (k + 1)])

        return ll.detach()