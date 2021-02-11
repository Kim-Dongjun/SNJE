import numpy as np
from loss.snl_loss import snl_loss
from loss.par_loss import par_loss

def loss(self, theta, x, mask, round, numEpoch):
    snl_loss_ = snl_loss(self, theta, x)
    if self.args.posteriorLearning:
        par_estimation = par_loss(self, theta, mask)
        c = np.log(9) / self.args.numRound
        self.lambda_ = 0.9 * np.exp(- round * c)
        c = np.log(50) / self.args.numRound
        self.lambda_ = 5. * np.exp(- round * c)
        return snl_loss_ + self.lambda_ * par_estimation
    else:
        return snl_loss_