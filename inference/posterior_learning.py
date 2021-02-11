import torch
from torch.nn.utils import clip_grad_norm_
from loss.apt_loss import apt_loss

def posterior_learning(self, round, numEpoch, sigma=0.01):
    self.netPosterior.train()
    if self.args.likelihoodLearning:
        self.netLikelihood.eval()
    for batch_idx, data in enumerate(self.train_loader_mask):

        training_theta = data[:,:self.args.thetaDim]
        training_x = data[:,self.args.thetaDim:self.args.thetaDim + self.args.xDim]
        training_mask = data[:,self.args.thetaDim + self.args.xDim:]
        if self.args.likelihoodLearning:
            self.optLikelihood.zero_grad()
        self.optPosterior.zero_grad()
        training_loss = apt_loss(self, round, training_theta, training_x, training_mask, self.args.use_non_atomic_loss).mean()
        training_loss.backward()
        if self.args.clip_grad_norm:
            clip_grad_norm_(
                self.netPosterior.parameters(), max_norm=1.,
            )
        self.optPosterior.step()

    self.netPosterior.eval()
    with torch.no_grad():
        self.current_epoch_validation_posterior_loss = apt_loss(self, round, self.validation_theta, self.validation_x).mean().item()

    if self.best_validation_posterior_loss > self.current_epoch_validation_posterior_loss:
        self.best_validation_posterior_loss = self.current_epoch_validation_posterior_loss
        self.validation_tolerance_posterior = 0
    else:
        self.validation_tolerance_posterior += 1
    print("Epoch " + str(numEpoch) + " neural posterior (training, validation) loss : ", training_loss.item(), self.current_epoch_validation_posterior_loss)
    if self.validation_tolerance_posterior >= self.args.maxValidationTolerance:
        return True
    else:
        return False