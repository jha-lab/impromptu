#monolithic deep viusla analogy making

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
import argparse

class EncoderCNN(nn.Module):
    def __init__(self, inp_size, hid_dim):
        super().__init__()
        """
        Downsample and embeds the input image into a latent space
        """
        self.conv1 = Conv2dBlock(3, hid_dim, 3,stride=2, padding = 1)
        self.conv2 = Conv2dBlock(hid_dim, hid_dim, 3,stride=2, padding = 1)
        self.conv3 = Conv2dBlock(hid_dim, hid_dim, 3,stride=2, padding = 1)
        self.conv4 = Conv2dBlock(hid_dim, hid_dim, 3,stride=2, padding = 1)
        self.feature_size = hid_dim*inp_size // 16 *inp_size // 16
        self.fc1 = linear(self.feature_size, hid_dim)
        self.fc2 = linear(hid_dim, hid_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class DecoderCNN(nn.Module):
    def __init__(self, inp_size, hid_dim):
        super().__init__()
        """
        Upsamples and decodes the latent space into an image
        """
        self.hid_dim = hid_dim
        self.inp_size = inp_size
        self.fc1 = linear(hid_dim, hid_dim)
        self.feature_size = hid_dim*inp_size // 16 *inp_size // 16
        self.fc2 = linear(hid_dim, self.feature_size)
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 3, stride=(2, 2), padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 3, stride=(2, 2), padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 3, stride=(2, 2), padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(hid_dim, 3, 3, stride=(2, 2), padding=1, output_padding=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = x.view(-1, self.hid_dim, self.inp_size // 16, self.inp_size // 16)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        return x

class TransformationNetwork(nn.Module):

    def __init__(self, hid_dim):
        super().__init__()
        """
        Makes the analogy in the latent space
        """
        self.fc1 = linear(2*hid_dim, hid_dim)
        self.fc2 = linear(hid_dim, hid_dim)
        self.fc3 = linear(hid_dim, hid_dim)
    
    def forward(self, context, query):

        mean_context = torch.mean(context, dim=1)
        proj = self.fc1(torch.cat((mean_context, query), dim=1))
        proj = F.relu(proj)
        proj = self.fc2(proj)
        proj = F.relu(proj)
        proj = self.fc3(proj)
        return proj


class InferenceNetwork(nn.Module):

    def __init__(self, hid_dim):
        super().__init__()
        """
        Makes the analogy in the latent space
        """
        self.fc1 = linear(2*hid_dim, hid_dim)
        self.fc2 = linear(hid_dim, hid_dim)
        self.fc3 = linear(hid_dim, hid_dim)
    
    def forward(self, support):

        B, num_examples, num_analogies, _ = support.shape
        support = support.reshape(B, num_examples, -1)
        context = self.fc1(support)
        context  = F.relu(context)
        context = self.fc2(context)
        context = F.relu(context)
        context = self.fc3(context)
        return context


class Monolithic(nn.Module):

    def __init__(self,args):

        super().__init__()
        self.d_model = args.d_model
        self.encoder = EncoderCNN(args.image_size, args.d_model)
        self.decoder = DecoderCNN(args.image_size, args.d_model)
        self.inference = InferenceNetwork(args.d_model)
        self.transformation = TransformationNetwork(args.d_model)
    
    def forward(self, support, query):
        
        B, N, _, C, H, W = support.shape
        support_embed = self.encoder(support.reshape(-1, C, H, W))
        query_embed = self.encoder(query.reshape(-1, C, H, W))
        support_embed = support_embed.reshape(B, N, -1, self.d_model)
        query_embed = query_embed.reshape(B, -1, self.d_model)
        context = self.inference(support_embed) 
        op = self.transformation(context, query_embed[:,0])
        recon = self.decoder(op)
        mse = ((query[:,1] - recon) ** 2).sum() / B

        return recon.clamp(0.,1.), mse

    def generate(self, support, query):

        B, N, _, C, H, W = support.shape
        support_embed = self.encoder(support.reshape(-1, C, H, W))
        query_embed = self.encoder(query)
        support_embed = support_embed.reshape(B, N, -1, self.d_model)
        context = self.inference(support_embed) 
        op = self.transformation(context, query_embed)
        recon = self.decoder(op)
        return recon.clamp(0.,1.)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=64)
    
    args = parser.parse_args()
    model = Monolithic(args)
    
    #unit test
    support = torch.randn(2, 9, 2, 3, 64, 64)
    query = torch.randn(2, 2, 3, 64, 64)
    recon, mse = model(support, query)
    print(recon.shape)
    print(mse)
    recon = model.generate(support, query[:,0])
    print(recon.shape)
    print("Unit test passed")


if __name__ == "__main__":

    main()