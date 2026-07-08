import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import OrderedDict
from utils.losses import l2_loss, l2_z_loss, ssim_loss, combined_loss, MorphologicalLoss, SummedSpatialLoss

from config.options import Options
from config import registry
opt = Options().parse()
registry.opt = opt

# -----------------------
# 1. Helper Functions
# -----------------------

def weights_init(mod: nn.Module) -> None:
    """
    Initializes weights of neural network layers.

    Applies:
        - Normal distribution for Conv and BatchNorm layers.
        - Xavier initialization for Linear layers.

    Args:
        mod (nn.Module): The module (layer) to initialize.
    """
    if hasattr(mod, 'weight'):
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(mod.weight.data, 0.0, 0.02)
        elif isinstance(mod, nn.BatchNorm2d):
            nn.init.normal_(mod.weight.data, 1.0, 0.02)
            if mod.bias is not None:
                nn.init.constant_(mod.bias.data, 0)
        elif isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight.data)
            if mod.bias is not None:
                nn.init.constant_(mod.bias.data, 0)

# -----------------------
# 2. Base Classes
# -----------------------

class BaseModel:
    """
    Base Model Class for managing seeds and devices.
    """
    def __init__(self, opt) -> None:
        self.seed(opt.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def seed(self, seed_value: int) -> None:
        """Sets seed for reproducibility."""
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

# -----------------------
# 3. Attention Mechanisms
# -----------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_layer_one = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU()
        )
        self.shared_layer_two = nn.Linear(in_channels // ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_out = self.shared_layer_one(self.avg_pool(x).view(b, c))
        avg_out = self.shared_layer_two(avg_out).view(b, c, 1, 1)

        max_out = self.shared_layer_one(self.max_pool(x).view(b, c))
        max_out = self.shared_layer_two(max_out).view(b, c, 1, 1)

        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
# -----------------------
# 4. Network Building Blocks
# -----------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cbam_flag):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbam = CBAMBlock(out_channels) if cbam_flag else None

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cbam_flag):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.cbam = CBAMBlock(out_channels) if cbam_flag else None

    def forward(self, x):
        x = self.up(x)
        x = self.conv_block(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x
# -----------------------
# 5. Main Architecture
# -----------------------

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, cbam_flag=True):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            EncoderBlock(in_channels, 8, cbam_flag),
            EncoderBlock(8, 16, cbam_flag),
            EncoderBlock(16, 32, cbam_flag),
            EncoderBlock(32, out_channels, cbam_flag),
        )

    def forward(self, x):
        return self.main(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, cbam_flag=False):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            DecoderBlock(in_channels, 64, cbam_flag),
            DecoderBlock(64, 32, cbam_flag),
            DecoderBlock(32, 16, cbam_flag),
            DecoderBlock(16, 8, cbam_flag),
        )
        self.final_layer = nn.Conv2d(8, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.main(x)
        x = self.final_layer(x)
        return self.activation(x)

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_cbam=True, decoder_cbam=False):
        super(Generator, self).__init__()

        self.encoder1 = Encoder(in_channels, out_channels, cbam_flag=encoder_cbam)
        self.decoder = Decoder(out_channels, in_channels, cbam_flag=decoder_cbam)
        self.encoder2 = Encoder(in_channels, out_channels, cbam_flag=encoder_cbam)

    def forward(self, x):
        latent_z = self.encoder1(x)
        fake = self.decoder(latent_z)
        latent_zhat = self.encoder2(fake)
        return fake, latent_z, latent_zhat

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = Encoder(1, 64)
        layers = list(model.main.children())
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        classification = self.classifier(features)
        return classification.view(-1, 1).squeeze(1), features

# -----------------------
# 6. GAN Class
# -----------------------

class GANLung(BaseModel):
    """
    Class for GANLung
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.generator = Generator(
            in_channels=1,
            out_channels=64,
            encoder_cbam = opt.model_architecture.encoder_cbam,
            decoder_cbam = opt.model_architecture.decoder_cbam
        ).to(self.device)

        self.discriminator = Discriminator().to(self.device)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.number = 00000000
        self.seed(opt.seed)

        self.l_adv = l2_loss

        if opt.con_loss == 'l2':
            self.l_con = l2_loss
        elif opt.con_loss == 'ssim':
            self.l_con = ssim_loss

        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()
        self.CombinedLoss = SummedSpatialLoss()
        # self.CombinedLoss = MorphologicalLoss()

        self.w_enc = opt.w_enc
        self.w_con = opt.w_con
        self.w_adv = opt.w_adv
        self.w_mmd = opt.w_mmd

        ## Optimizers
        self.generator.train()
        self.discriminator.train()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
    def compute_kernel(self, x, y, sigma=1.0):
        x = x.unsqueeze(1)  # (B, 1, D)
        y = y.unsqueeze(0)  # (1, B, D)
        dist = ((x - y) ** 2).sum(2)  # (B, B)
        return torch.exp(-dist / (2 * sigma ** 2))

    def compute_mmd(self, x, y, sigma=1.0):
        K_xx = self.compute_kernel(x, x, sigma)
        K_yy = self.compute_kernel(y, y, sigma)
        K_xy = self.compute_kernel(x, y, sigma)

        return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


    def forward_g(self, input):
        """ Forward propagate through Generator """
        fake, latent_i, latent_o = self.generator(input)
        return fake, latent_i, latent_o

    def forward_d(self, input, fake):
        """ Forward propagate through Discriminator """
        pred_real, feat_real = self.discriminator(input)
        pred_fake, feat_fake = self.discriminator(fake.detach())
        return pred_real, feat_real, pred_fake, feat_fake

    def backward_g(self, input, fake, latent_i, latent_o):
        """ Backpropagate through Generator """
        err_g_adv = self.l_adv(self.discriminator(input)[1], self.discriminator(fake)[1])
        err_g_con = self.CombinedLoss(fake, input)
        err_g_enc = self.l_enc(latent_o, latent_i)
        mmd = self.compute_mmd(latent_i, latent_o)

        err_g = (
            err_g_adv * self.w_adv +
            err_g_con * self.w_con +
            err_g_enc * self.w_enc +
            mmd * self.w_mmd
        )
        err_g.backward(retain_graph=True)

        return err_g, err_g_con, err_g_adv, err_g_enc, mmd


    def backward_d(self, pred_real, pred_fake):
        """ Backpropagate through Discriminator """
        # Real - Fake Loss
        real_label = torch.ones_like(pred_real, device=self.device)
        fake_label = torch.zeros_like(pred_fake, device=self.device)

        err_d_real = self.l_bce(pred_real, real_label)
        err_d_fake = self.l_bce(pred_fake, fake_label)

        # Discriminator Loss & Backward-Pass
        err_d = (err_d_real + err_d_fake) * 0.5
        err_d.backward()
        return err_d

    def reinit_d(self):
        """ Re-initialize the weights of netD """
        self.discriminator.apply(weights_init)
        print('   Reloading net d')

    def optimize_params(self, input):
        """ Forwardpass, Loss Computation and Backwardpass. """
        # Forward-pass
        fake, latent_i, latent_o = self.forward_g(input)
        pred_real, feat_real, pred_fake, feat_fake = self.forward_d(input, fake)

        # Backward-pass for generator
        self.optimizer_G.zero_grad()
        err_g, err_g_con, err_g_adv, err_g_enc, mmd = self.backward_g(input, fake, latent_i, latent_o)
        self.optimizer_G.step()

        # Backward-pass for discriminator
        self.optimizer_D.zero_grad()
        err_d = self.backward_d(pred_real, pred_fake)
        self.optimizer_D.step()
    
        if err_d.item() < 1e-5:
            self.reinit_d()

        return err_g, err_d, err_g_con, err_g_adv, err_g_enc, mmd

    def save_model(self, save_path, model_name='ganlung'):
        # Ensure the save path is absolute based on PYTHONPATH if available
        save_path = os.path.abspath(save_path)
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict()
        }, os.path.join(save_path, f'{model_name}.pth'))  # Use model_name from options


    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])