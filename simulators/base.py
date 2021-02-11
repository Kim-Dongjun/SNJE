import numpy as np
import torch
import torch.distributions
from utils.input_range import get_input_range
from simulators.simulator import get_simulator

class Base():

    def __init__(self, args):
        self.args = args
        self.min, self.max = get_input_range(args)
        self.simulator = get_simulator(args)

    def get_simulation_result(self, thetas, observation=False):

        return self.simulator.run(thetas, observation)
