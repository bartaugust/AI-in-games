import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class GAN:
    def __init__(self, cfg):
        super().__init__()
        self.sample_dir = 'generated'
        self.stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        os.makedirs(self.sample_dir, exist_ok=True)
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        def build_discriminator():
            discriminator = nn.Sequential(
                # in: 3 x 64 x 64

                nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 64 x 32 x 32

                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 128 x 16 x 16

                nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 256 x 8 x 8

                nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(2048),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 512 x 4 x 4

                nn.Conv2d(2048, 1, kernel_size=4, stride=1, padding=0, bias=False),
                # out: 1 x 1 x 1

                nn.Flatten(),
                nn.Sigmoid())

            return discriminator

        def build_generator():
            generator = nn.Sequential(
                # in: latent_size x 1 x 1

                nn.ConvTranspose2d(self.cfg.model.latent_size, 2048, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(True),
                # out: 512 x 4 x 4

                nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(True),
                # out: 256 x 8 x 8

                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # out: 128 x 16 x 16

                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # out: 64 x 32 x 32

                nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()
                # out: 3 x 64 x 64
            )

            return generator

        self.discriminator = build_discriminator().to(self.device)
        self.generator = build_generator().to(self.device)

    def denorm(self, img_tensors):
        return img_tensors * self.stats[1][0] + self.stats[0][0]

    def train_discriminator(self, real_images, opt_d):
        # Clear discriminator gradients
        opt_d.zero_grad()
        real_images = real_images.to(self.device)
        # Pass real images through discriminator
        real_preds = self.discriminator(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=self.device)
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        # Generate fake images
        latent = torch.randn(self.cfg.model.batch_size, self.cfg.model.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=self.device)
        fake_preds = self.discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        opt_d.step()
        return loss.item(), real_score, fake_score

    def train_generator(self, opt_g):
        # Clear generator gradients
        opt_g.zero_grad()

        # Generate fake images
        latent = torch.randn(self.cfg.model.batch_size, self.cfg.model.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)

        # Try to fool the discriminator
        preds = self.discriminator(fake_images)
        targets = torch.ones(self.cfg.model.batch_size, 1, device=self.device)
        loss = F.binary_cross_entropy(preds, targets)

        # Update generator weights
        loss.backward()
        opt_g.step()

        return loss.item()

    def save_samples(self, index, latent_tensors, show=True):
        fake_images = self.generator(latent_tensors)
        fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
        save_image(self.denorm(fake_images), os.path.join(self.sample_dir, fake_fname), nrow=8)
        print('Saving', fake_fname)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]);
            ax.set_yticks([])
            ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

    def fit(self, train_dl, start_idx=1):
        torch.cuda.empty_cache()

        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []

        # Create optimizers
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.model.lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.cfg.model.lr, betas=(0.5, 0.999))

        for epoch in range(self.cfg.model.epochs):
            for real_images, _ in tqdm(train_dl):
                # Train discriminator
                loss_d, real_score, fake_score = self.train_discriminator(real_images, opt_d)
                # Train generator
                loss_g = self.train_generator(opt_g)

            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)

            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch + 1, self.cfg.model.epochs, loss_g, loss_d, real_score, fake_score))

            # Save generated images
            # save_samples(epoch + start_idx, fixed_latent, show=False)

        return losses_g, losses_d, real_scores, fake_scores
