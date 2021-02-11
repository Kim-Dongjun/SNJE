import argparse



def get_parser():
	parser = argparse.ArgumentParser(description='')

	# Simulation settings
	parser.add_argument("--simulation", default='SLCP-16', help="SLCP-16/SLCP-256/MG1/Ricker/Poisson/MNIST/FashionMNIST/SVHN")
	parser.add_argument("--thetaDim", type=int, default=5, help="Simulation Input dimension")
	parser.add_argument("--xDim", type=int, default=50, help="Simulation output dimension")
	parser.add_argument("--numModes", type=int, default=16, help="Number of posterior modes")
	parser.add_argument("--numTime", type=int, default=1000, help="Simulation execution timestep")

	# Inference settings
	parser.add_argument("--numRound", type=int, default=30, help="Number of rounds for the inference")
	parser.add_argument("--simulation_budget_per_round", type=int, default=100, help="Number of simulations per a round")

	# put --LikelihoodLearning True --posteriorLearning True in command line if you watn to learn SNL with PAR
	parser.add_argument("--posteriorLearning", type=bool, default=False, help='True if you want to learn APT')
	parser.add_argument("--likelihoodLearning", type=bool, default=False, help='True if you watn to learn SNL')


	# Learning settings
	parser.add_argument("--use_non_atomic_loss", type=bool, default=False,
						help='Whether to use Mixuter of Gaussian for the proposal distribution in APT')
	parser.add_argument("--clip_grad_norm", type=bool, default=True)
	parser.add_argument("--lrLikelihood", type=float, default=1e-3, help="Learning rate of likelihood estimation")
	parser.add_argument("--lrPosterior", type=float, default=5e-4, help="Learning rate of posterior estimation")
	parser.add_argument("--nsfTailBound", type=float, default=10.0, help="Neural Spline Flow tail bound")
	parser.add_argument("--validationRatio", type=float, default=0.1, help="Validation dataset ratio")
	parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
	parser.add_argument("--numMaxEpoch", type=int, default=300, help='No learning more than numMaxEpoch epochs')
	parser.add_argument("--maxValidationTolerance", type=int, default=20,
						help="Maximum epochs that validation loss does not minimized anymore")

	# Network settings
	parser.add_argument("--likelihoodHiddenDim", type=int, default=32, help="Likelihood hidden layer dimension")
	parser.add_argument("--posteriorHiddenDim", type=int, default=32, help="Posterior hidden layer dimension")
	parser.add_argument("--likelihoodNumBlocks", type=int, default=4,
						help="Number of blocks of flow model used in likelihood estimation")
	parser.add_argument("--posteriorNumBlocks", type=int, default=4,
						help="Number of blocks of flow model used in posterior estimation")
	parser.add_argument("--likelihoodNumBin", type=int, default=64, help="Number of bins for likelihood network")
	parser.add_argument("--posteriorNumBin", type=int, default=64, help="Number of bins for posterior network")

	# Sampler settings
	parser.add_argument("--burnInMCMC", type=int, default=200, help="Number of burn-in periods for MCMC algorithm")

	# Device setting
	parser.add_argument("--device", default='cuda:0', help="Device to update parameters")
	args = parser.parse_args()

	return args