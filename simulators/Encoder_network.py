import torch.nn.functional as F
import torch

class ConvVariationalAutoencoder(torch.nn.Module):

    def __init__(self, device, in_channels, num_features, num_latent):
        super(ConvVariationalAutoencoder, self).__init__()

        ###############
        # ENCODER
        ##############
        self.device = device

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        self.enc_conv_1 = torch.nn.Conv2d(in_channels=in_channels,
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

        self.z_mean = torch.nn.Linear(64 * 2 * 2, num_latent)
        # in the original paper (Kingma & Welling 2015, we use
        # have a z_mean and z_var, but the problem is that
        # the z_var can be negative, which would cause issues
        # in the log later. Hence we assume that latent vector
        # has a z_mean and z_log_var component, and when we need
        # the regular variance or std_dev, we simply use
        # an exponential function
        self.z_log_var = torch.nn.Linear(64 * 2 * 2, num_latent)

        ###############
        # DECODER
        ##############

        self.dec_linear_1 = torch.nn.Linear(num_latent, 64 * 2 * 2)

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
                                                     out_channels=in_channels,
                                                     kernel_size=(6, 6),
                                                     stride=(3, 3),
                                                     padding=4)

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def encoder(self, features):
        print("feature shape : ", features.shape)
        x = self.enc_conv_1(features)
        x = F.leaky_relu(x)
        # print('conv1 out:', x.size())

        x = self.enc_conv_2(x)
        x = F.leaky_relu(x)
        # print('conv2 out:', x.size())

        x = self.enc_conv_3(x)
        x = F.leaky_relu(x)
        # print('conv3 out:', x.size())

        z_mean = self.z_mean(x.view(-1, 64 * 2 * 2))
        z_log_var = self.z_log_var(x.view(-1, 64 * 2 * 2))
        encoded = self.reparameterize(z_mean, z_log_var)

        return z_mean, z_log_var, encoded

    def decoder(self, encoded):
        x = self.dec_linear_1(encoded)
        x = x.view(-1, 64, 2, 2)

        x = self.dec_deconv_1(x)
        x = F.leaky_relu(x)
        # print('deconv1 out:', x.size())

        x = self.dec_deconv_2(x)
        x = F.leaky_relu(x)
        # print('deconv2 out:', x.size())

        x = self.dec_deconv_3(x)
        x = F.leaky_relu(x)
        # print('deconv1 out:', x.size())

        decoded = torch.sigmoid(x)
        print("decoded shape : ", decoded.shape)
        return decoded

    def forward(self, features):
        z_mean, z_log_var, encoded = self.encoder(features)
        decoded = self.decoder(encoded)

        return z_mean, z_log_var, encoded, decoded

