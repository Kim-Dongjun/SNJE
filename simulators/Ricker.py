import torch
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


class Ricker():
    def __init__(self, args):
        self.args = args

    def run(self, thetas, observation):

        log_n_t = torch.Tensor([0.] * thetas.shape[0]).reshape(-1, thetas.shape[0]).to(self.args.device)
        result = torch.zeros((len(thetas), self.args.numTime + 1)).to(self.args.device)
        thetas_t = torch.t(thetas)

        for t in range(self.args.numTime):
            log_n_t = torch.cat((log_n_t, (thetas_t[0] + log_n_t[-1] - torch.exp(log_n_t[-1]) + thetas_t[1] * torch.randn(
                (thetas.shape[0])).to(self.args.device)).reshape(1, -1)))
        log_n_t = torch.t(log_n_t)

        for i in range(thetas.shape[0]):
            for j in range(self.args.numTime + 1):
                result[i][j] = torch.distributions.poisson.Poisson(thetas_t[2][i] * torch.exp(log_n_t[i][j])).sample()

        res = result[:, 1:].detach()

        if self.args.xDim == 100:
            return res

        elif self.args.xDim == 20:
            return torch.t(torch.Tensor(np.quantile(res.t().cpu().detach().numpy(),
                                                    (1. / (self.args.xDim - 1)) * np.arange(self.args.xDim),
                                                    axis=0)).to(self.args.device)).detach()

        elif self.args.xDim == 13:
            return self.get_summary_statistics(res)

        else:
            assert self.args.xDim == res.shape[1]

    def get_summary_statistics(self, sim_output):
        sumstats = torch.Tensor([[]]).to(self.args.device)
        sumstats = torch.cat((sumstats, torch.mean(sim_output, axis=1).reshape(1, -1).to(self.args.device)), dim=1)
        sumstats = torch.cat(
            (sumstats, torch.sum(sim_output == 0, axis=1, dtype=torch.float).reshape(1, -1).to(self.args.device)))

        def autocov(x, lag=1):
            C = torch.mean(x[:, lag:] * x[:, :-lag], axis=1) - torch.mean(x[:, lag:], axis=1) * torch.mean(x[:, :-lag],
                                                                                                           axis=1)
            return C.reshape(1, -1)

        for k in range(5):
            sumstats = torch.cat((sumstats, autocov(sim_output, lag=k + 1).to(self.args.device)))

        def funct(x, a, b, c):
            return a + b * x + c * x ** 2

        betas = np.zeros((2, len(sim_output)))
        sim_output_ = sim_output.cpu().detach().numpy()
        x = np.power(sim_output_, 0.3)
        for batch in range(len(sim_output)):
            input = x[batch][:-1]
            y = x[batch][1:]
            # print("input : ", input)
            # print("y : ", y)
            popt_cons, _ = curve_fit(funct, input, y, bounds=([0, -np.inf, -np.inf], [0.000000001, np.inf, np.inf]))
            betas[0][batch] = popt_cons[1]
            betas[1][batch] = popt_cons[2]
        for k in range(2):
            sumstats = torch.cat((sumstats, torch.Tensor(betas[k]).to(self.args.device).reshape(1, -1)))

        gammas = np.zeros((4, len(sim_output)))
        for batch in range(len(sim_output)):
            input = sim_output_[batch][1:]
            y = sim_output_[batch][1:] - sim_output_[batch][:-1]
            polynomial_features = PolynomialFeatures(degree=3)
            x_poly = polynomial_features.fit_transform(input.reshape(-1, 1))
            model_ = LinearRegression()
            model_.fit(x_poly, y)
            coe = model_.coef_.reshape(-1)
            gammas[0][batch] = coe[0]
            gammas[1][batch] = coe[1]
            gammas[2][batch] = coe[2]
            gammas[3][batch] = coe[3]
        for k in range(4):
            sumstats = torch.cat((sumstats, torch.Tensor(gammas[k]).to(self.args.device).reshape(1, -1)))

        return torch.t(sumstats)