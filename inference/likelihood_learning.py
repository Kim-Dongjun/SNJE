from loss.loss import loss
from torch.nn.utils import clip_grad_norm_
import torch

def likelihood_learning(self, round, numEpoch):
    self.netLikelihood.train()
    if self.args.posteriorLearning:
        self.netPosterior.eval()

    for batch_idx, data in enumerate(self.train_loader_mask):
        training_theta = data[:, :self.args.thetaDim]
        training_x = data[:, self.args.thetaDim:self.args.thetaDim + self.args.xDim]
        training_mask = data[:, self.args.thetaDim + self.args.xDim:]
        self.optLikelihood.zero_grad()
        training_loss = loss(self, training_theta, training_x, training_mask, round, numEpoch)
        training_loss.backward()
        if self.args.clip_grad_norm:
            clip_grad_norm_(
                self.netLikelihood.parameters(), max_norm=5.,)
        self.optLikelihood.step()

    self.netLikelihood.eval()
    with torch.no_grad():
        self.current_epoch_validation_loss = loss(self, self.validation_theta,
                              self.validation_x, self.validation_mask, round, numEpoch).item()

    print("Epoch " + str(numEpoch) + " neural likelihood (training, validation) loss : ", training_loss.item(), self.current_epoch_validation_loss)

    if self.best_validation_loss > self.current_epoch_validation_loss:
        self.best_validation_loss = self.current_epoch_validation_loss
        self.validation_tolerance_likelihood = 0
    else:
        self.validation_tolerance_likelihood += 1

    if self.validation_tolerance_likelihood >= self.args.maxValidationTolerance:
        return True
    else:
        return False