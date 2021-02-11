import torch
from neural_nets.nsf import neural_net_nsf

def neural_posterior(self):

    def posteriorNetwork(training_theta=None, training_x=None):

        bounds = torch.max(torch.max(torch.abs(self.sim.max)),torch.max(torch.abs(self.sim.min))).item()

        netPosterior = neural_net_nsf(self, self.args.posteriorHiddenDim, self.args.posteriorNumBlocks, self.args.posteriorNumBin,
                                      self.args.thetaDim, self.args.xDim, batch_x=training_theta, batch_theta=training_x, tail=self.args.nsfTailBound, bounded=True).to(self.args.device)

        optimizer = torch.optim.Adam(netPosterior.parameters(), lr=self.args.lrPosterior, betas=(0.5, 0.999), weight_decay=1e-5)

        return netPosterior, optimizer

    return posteriorNetwork