from utils.parser import get_parser
from simulators.base import Base
from utils.true_input import get_true_input
from neural_nets.neural_likelihood import neural_likelihood
from neural_nets.neural_posterior import neural_posterior
from sampler.sample_from_neural_likelihood import sample_from_likelihood
from sampler.sample_from_neural_posterior import sample_from_posterior
from inference.train import train
from sampler.prior import Prior

class main():
    def __init__(self, args):
        self.args = args

        # Define simulation engine
        self.sim = Base(self.args)

        # Get true parameter
        self.true_input = get_true_input(self.args)

        # Get observation
        self.observation = self.sim.get_simulation_result(self.true_input, True)

        # Define neural networks
        if self.args.likelihoodLearning:
            self.likelihood = neural_likelihood(self)
        if self.args.posteriorLearning:
            self.posterior = neural_posterior(self)

        # Define prior
        self.prior = Prior(self.sim)

        # Main loop
        for round in range(self.args.numRound):
            print(str(round)+"-th round of training start")
            self.train(round)

    def train(self, round):

        # Batch of simulation input sampling
        if self.args.likelihoodLearning:
            self.thetas = sample_from_likelihood(self, self.args.simulation_budget_per_round, round == 0).detach().to(self.args.device)
        else:
            self.thetas = sample_from_posterior(self, self.args.simulation_budget_per_round, round == 0).detach().to(self.args.device)

        # Simulation execution
        self.simulated_output = self.sim.get_simulation_result(self.thetas)
        print("simulation input, output shape : ", self.thetas.shape, self.simulated_output.shape)

        # Posterior inference
        train(self, round)

if __name__ == "__main__":
    args = get_parser()
    inference = main(args)