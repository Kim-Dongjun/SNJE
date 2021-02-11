import torch
from inference.likelihood_learning import likelihood_learning
from inference.posterior_learning import posterior_learning


def train(self, round):
    # Put simulation input and output into data repository
    permutation = torch.randperm(self.args.simulation_budget_per_round)
    if round == 0:
        permutation = torch.randperm(self.thetas.shape[0])
        self.training_theta = self.thetas[
            permutation[int(self.args.validationRatio * self.thetas.shape[0]):]]
        self.training_x = self.simulated_output[
            permutation[int(self.args.validationRatio * self.thetas.shape[0]):]]
        self.training_mask = torch.ones((self.training_theta.shape[0], 1)).to(self.args.device)
        self.validation_theta = self.thetas[
            permutation[:int(self.args.validationRatio * self.thetas.shape[0])]]
        self.validation_x = self.simulated_output[
            permutation[:int(self.args.validationRatio * self.thetas.shape[0])]]
        self.validation_mask = torch.ones((self.validation_theta.shape[0], 1)).to(self.args.device)
        self.validation_fake_x = self.validation_x
        if self.args.likelihoodLearning:
            self.netLikelihood, self.optLikelihood = self.likelihood(training_theta=self.training_theta, training_x=self.training_x)
        if self.args.posteriorLearning:
            self.netPosterior, self.optPosterior = self.posterior(training_theta=self.training_theta, training_x=self.training_x)
    elif round > 0:
        self.training_theta = torch.cat((self.training_theta, self.thetas[
            permutation[int(self.args.validationRatio * self.args.simulation_budget_per_round):]]))
        self.training_x = torch.cat((self.training_x, self.simulated_output[
            permutation[int(self.args.validationRatio * self.args.simulation_budget_per_round):]]))
        self.training_mask = torch.cat((self.training_mask, torch.zeros((int((1 - self.args.validationRatio) * self.thetas.shape[0]),1)).to(self.args.device))).to(self.args.device)
        self.validation_theta = torch.cat((self.validation_theta, self.thetas[
            permutation[:int(self.args.validationRatio * self.args.simulation_budget_per_round)]]))
        self.validation_x = torch.cat((self.validation_x, self.simulated_output[
            permutation[:int(self.args.validationRatio * self.args.simulation_budget_per_round)]]))
        self.validation_mask = torch.cat((self.validation_mask, torch.zeros(
            (int(self.args.validationRatio * self.thetas.shape[0]), 1)).to(self.args.device))).to(
            self.args.device)
        self.validation_fake_x = self.validation_x

    # Make train loader for learning
    self.train_loader_mask = torch.utils.data.DataLoader(torch.cat((self.training_theta, self.training_x, self.training_mask), 1),
                                                         batch_size=self.args.batch_size, shuffle=True)

    # Learning setup
    converged = False
    numEpoch = 0
    self.best_validation_loss = 1e10
    self.best_validation_posterior_loss = 1e10
    self.validation_tolerance_likelihood = 0
    self.validation_tolerance_posterior = 0

    # Learning by early stopping
    while not converged:

        if self.args.posteriorLearning:
            convergedPosterior = posterior_learning(self, round, numEpoch)
        else:
            convergedPosterior = True

        if self.args.likelihoodLearning:
            convergedLikelihood = likelihood_learning(self, round, numEpoch)
        else:
            convergedLikelihood = True

        if self.args.likelihoodLearning:
            if convergedLikelihood:
                converged = True
        else:
            if convergedPosterior == True:
                converged = True

        numEpoch += 1
        if numEpoch == self.args.numMaxEpoch:
            break
    if self.args.likelihoodLearning:
        self.netLikelihood.eval()
    if self.args.posteriorLearning:
        self.netPosterior.eval()