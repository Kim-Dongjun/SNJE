import numpy as np
import torch
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from itertools import chain
from numpy import prod

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class summary_statistics(torch.nn.Module):
    def __init__(self, xDim):
        super(summary_statistics, self).__init__()

        ###############
        # ENCODER
        ##############

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        self.xDim = xDim

        self.enc_conv_1 = torch.nn.Conv2d(in_channels=1,
                                          out_channels=16,
                                          kernel_size=(6, 6),
                                          stride=(2, 2),
                                          padding=0)

        self.enc_conv_2 = torch.nn.Conv2d(in_channels=16,
                                          out_channels=32,
                                          kernel_size=(4, 4),
                                          stride=(2, 2),
                                          padding=0)

        self.enc_conv_3 = torch.nn.Conv2d(in_channels=32,
                                          out_channels=64,
                                          kernel_size=(2, 2),
                                          stride=(2, 2),
                                          padding=0)

        self.z_mean = torch.nn.Linear(64 * 2 * 2, self.xDim)
        # in the original paper (Kingma & Welling 2015, we use
        # have a z_mean and z_var, but the problem is that
        # the z_var can be negative, which would cause issues
        # in the log later. Hence we assume that latent vector
        # has a z_mean and z_log_var component, and when we need
        # the regular variance or std_dev, we simply use
        # an exponential function
        self.z_log_var = torch.nn.Linear(64 * 2 * 2, self.xDim)

        ###############
        # DECODER
        ##############

        self.dec_linear_1 = torch.nn.Linear(self.xDim, 64 * 2 * 2)

        self.dec_deconv_1 = torch.nn.ConvTranspose2d(in_channels=64,
                                                     out_channels=32,
                                                     kernel_size=(2, 2),
                                                     stride=(2, 2),
                                                     padding=0)

        self.dec_deconv_2 = torch.nn.ConvTranspose2d(in_channels=32,
                                                     out_channels=16,
                                                     kernel_size=(4, 4),
                                                     stride=(3, 3),
                                                     padding=1)

        self.dec_deconv_3 = torch.nn.ConvTranspose2d(in_channels=16,
                                                     out_channels=1,
                                                     kernel_size=(6, 6),
                                                     stride=(3, 3),
                                                     padding=4)

    def get_summary_statistics_mean(self, features):
        x = self.enc_conv_1(features)
        x = F.leaky_relu(x)
        # print('conv1 out:', x.size())

        x = self.enc_conv_2(x)
        x = F.leaky_relu(x)
        # print('conv2 out:', x.size())

        x = self.enc_conv_3(x)
        x = F.leaky_relu(x)
        # print('conv3 out:', x.size())

        representation = self.z_mean(x.view(-1, 64 * 2 * 2))

        return representation.reshape(-1,self.xDim)

    def get_encoded_mean_logvar(self, features):
        x = self.enc_conv_1(features)
        x = F.leaky_relu(x)
        # print('conv1 out:', x.size())

        x = self.enc_conv_2(x)
        x = F.leaky_relu(x)
        # print('conv2 out:', x.size())

        x = self.enc_conv_3(x)
        x = F.leaky_relu(x)
        # print('conv3 out:', x.size())

        representation = torch.cat((self.z_mean(x.view(-1, 64 * 2 * 2)), self.z_log_var(x.view(-1, 64 * 2 * 2))), 1)
        #print(self.z_mean(x.view(-1, 64 * 2 * 2)).shape, self.z_log_var(x.view(-1, 64 * 2 * 2)).shape, representation.shape)
        return representation#.reshape(-1,2 * self.xDim)

class Encoder(nn.Module):
    """
    Encoder, x --> mu, log_sigma_sq
    """

    def __init__(self, xDim):
        super(Encoder, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )
        self.linear_mu = nn.Linear(512, xDim)
        self.linear_log_sigma_sq = nn.Linear(512, xDim)
        self.reset_bias_and_weights()

    def reset_bias_and_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        h = self.main(input)
        h = h.resize(h.size(0), h.size(1) * h.size(2) * h.size(3))
        return self.linear_mu(h), self.linear_log_sigma_sq(h)

class Decoder(nn.Module):
    """
    Decoder, N(mu, log_sigma_sq) --> z --> x
    """

    def __init__(self, xDim):
        super(Decoder, self).__init__()
        self.main_1 = nn.Sequential(
            nn.Linear(xDim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.main_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )
        self.reset_bias_and_weights()

    def reset_bias_and_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def forward(self, input):
        h = self.main_1(input)
        h = h.resize(input.size(0), 512)
        x = self.main_2(h)
        return x

class summary_statistics_svhn(nn.Module):
    """
    VAE, x --> mu, log_sigma_sq --> N(mu, log_sigma_sq) --> z --> x
    """

    def __init__(self, xDim):
        super(summary_statistics_svhn, self).__init__()
        self.xDim = xDim
        self.Encoder = Encoder(self.xDim)
        self.Decoder = Decoder(self.xDim)

    def get_encoded_mean_logvar(self, input):
        self.mu, self.log_sigma_sq = self.Encoder(input)
        res = torch.cat((self.mu, self.log_sigma_sq), 1)
        return res


class Image():
    def __init__(self, args):
        self.args = args
        self.generator, self.summary_statistics = self.generator_buildup()

    def generator_buildup(self):
        if self.args.simulation == 'MNIST' or self.args.simulation == 'FashionMNIST':
            model = generator(input_dim=self.args.thetaDim, output_dim=1, input_size=28)
        elif self.args.simulation == 'SVHN':
            model = generator(input_dim=self.args.thetaDim, output_dim=3, input_size=32)
        if self.args.simulation == 'MNIST':
            model.load_state_dict(torch.load(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/generator_parameters/MNIST_WGAN_GP_Generator.pkl'))
        elif self.args.simulation == 'FashionMNIST':
            model.load_state_dict(torch.load(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))) + '/generator_parameters/FASHION-MNIST_WGAN_GP_Generator.pkl'))
        elif self.args.simulation == 'SVHN':
            model.load_state_dict(torch.load(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))) + '/generator_parameters/SVHN_WGAN_GP_Generator.pkl'))
        model.to(self.args.device)
        model.eval()

        if self.args.simulation == 'MNIST':
            ss = summary_statistics(self.args.xDim)
            ss.load_state_dict(torch.load(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/generator_parameters/MNIST_Summary_Statistics.pt'))
        elif self.args.simulation == 'FashionMNIST':
            ss = summary_statistics(self.args.xDim // 2)
            ss.load_state_dict(torch.load(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))) + '/generator_parameters/FASHION-MNIST_Summary_Statistics.pt'))
        elif self.args.simulation == 'SVHN':
            ss = summary_statistics_svhn(self.args.xDim // 2)
            ss.load_state_dict(torch.load(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))) + '/generator_parameters/SVHN_Summary_Statistics.pth'))
        ss.to(self.args.device)
        ss.eval()

        return model, ss

    def run(self, thetas):
        image = self.generator(thetas)
        if self.args.simulation == 'MNIST':
            return self.summary_statistics.get_encoded_mean(image.reshape(thetas.shape[0], 1, 28, 28).detach()).detach()
        elif self.args.simulation == 'FashionMNIST':
            return self.summary_statistics.get_encoded_mean_logvar(
                image.reshape(thetas.shape[0], 1, 28, 28).detach()).detach()
        elif self.args.simulation == 'SVHN':
            return self.summary_statistics.get_encoded_mean_logvar(
                image.reshape(thetas.shape[0], 3, 32, 32).detach()).detach()