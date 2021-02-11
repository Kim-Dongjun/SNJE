import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np

def plot_marginal_distribution(self, round, input_theta, title, upper=True):
    plt.close()
    fig = plt.figure(figsize=(16 ,16))
    for i in range(self.args.thetaDim):
        for j in range(i, self.args.thetaDim) if upper else range(i + 1):

            ax = fig.add_subplot(self.args.thetaDim, self.args.thetaDim, i * self.args.thetaDim + j + 1)

            if i == j:
                bandwidths = 10 ** np.linspace(-4, 0, 30)
                grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                    {'bandwidth': bandwidths},
                                    cv=10)
                grid.fit(input_theta[: ,j].reshape(-1 ,1))
                kde = grid.best_estimator_
                likelihood = np.exp(kde.score_samples(self.bases[j])).reshape(-1)
                ax.plot(self.bases[j], likelihood)
                ax.set_xlim(self.thetaDomain[j])
                ax.set_ylim([0.0, ax.get_ylim()[1]])
                ax.set_ylim([0.0, ax.get_ylim()[1]])
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                if i < self.args.thetaDim - 1 and not upper:
                    ax.tick_params(axis='x', which='both', labelbottom=False)
                if self.true_thetas.shape[0] > 1:
                    for k in range(self.true_thetas.shape[0]):
                        ax.vlines(self.true_thetas[k][j].item(), 0, ax.get_ylim()[1], color='r')
                else:
                    if self.parent.true_theta is not None: ax.vlines(self.parent.true_theta[0][j].item(), 0, ax.get_ylim()[1], color='r')

            else:
                ax.scatter(input_theta[: ,j], input_theta[: ,i], s=1, color='black')
                ax.set_xlim(self.thetaDomain[j])
                ax.set_ylim(self.thetaDomain[i])
                if i < self.args.thetaDim - 1: ax.tick_params(axis='x', which='both', labelbottom=False)
                if j > 0: ax.tick_params(axis='y', which='both', labelleft=False)
                if j == self.args.thetaDim - 1: ax.tick_params(axis='y', which='both', labelright=True)
                if self.true_thetas.shape[0] > 1:
                    for k in range(self.true_thetas.shape[0]):
                        ax.plot(self.true_thetas[k][j].item(), self.true_thetas[k][i].item(), 'r.', ms=8)
                else:
                    if self.parent.true_theta is not None: ax.plot(self.parent.true_theta[0][j].item(), self.parent.true_theta[0][i].item(), 'r.', ms=8)
    plt.tight_layout()
    plt.savefig(self.dir + '/samples_from_' + str(title) + '_round_' + str(round) + '.png')