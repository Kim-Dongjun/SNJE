import torch
from neural_nets.nsf import neural_net_nsf

def neural_likelihood(self):

    def likelihoodNetwork(training_theta=None, training_x=None):

        netLikelihood = neural_net_nsf(self, self.args.likelihoodHiddenDim, self.args.likelihoodNumBlocks, self.args.likelihoodNumBin,
                                       self.args.xDim, self.args.thetaDim, batch_x=training_x, batch_theta=training_theta, tail=self.args.nsfTailBound).to(self.args.device)

        optimizer = torch.optim.Adam(netLikelihood.parameters(), lr=self.args.lrLikelihood, betas=(0.5, 0.999), weight_decay=1e-5)

        return netLikelihood, optimizer

    return likelihoodNetwork