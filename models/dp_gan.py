import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


class DP_GAN:
    def __init__(
            self,
            generator,
            discriminator,
            latent_dim,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def train(self, dataloader, optimizer_G, optimizer_D, criterion=nn.BCELoss(), epochs=20, device='cpu'):
        history = {'G_loss': [], 'D_loss': [], 'samples': []}

        for epoch in range(epochs):
            G_loss_total = 0.0
            D_loss_total = 0.0

            for real_data, _ in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
                batch_size = real_data.size(0)
                real_data = real_data.to(device)
                real_labels = torch.ones(batch_size, 1, device=device) * 0.9  # Label smoothing
                fake_labels = torch.zeros(batch_size, 1, device=device)

                # Train Discriminator
                optimizer_D.zero_grad()
                noise = torch.randn(batch_size, self.latent_dim, device=device)
                fake_data = self.generator(noise)
                real_output = self.discriminator(real_data)
                fake_output = self.discriminator(fake_data.detach())
                real_loss = criterion(real_output, real_labels)
                fake_loss = criterion(fake_output, fake_labels)
                D_loss = real_loss + fake_loss
                D_loss.backward()
                optimizer_D.step()

                # Train Generator
                optimizer_G.zero_grad()
                noise = torch.randn(batch_size, self.latent_dim, device=device)
                fake_data = self.generator(noise)
                output = self.discriminator(fake_data)
                G_loss = criterion(output, real_labels)
                G_loss.backward()
                optimizer_G.step()

                G_loss_total += G_loss.item()
                D_loss_total += D_loss.item()

            avg_G_loss = G_loss_total / len(dataloader)
            avg_D_loss = D_loss_total / len(dataloader)
            history['G_loss'].append(avg_G_loss)
            history['D_loss'].append(avg_D_loss)

            print(f"Epoch [{epoch + 1}/{epochs}], Generator Loss: {avg_G_loss}, Discriminator Loss: {avg_D_loss}")

            if epoch % 10 == 0 or epoch == epochs - 1:
                noise = torch.randn(16, self.latent_dim, device=device)
                samples = self.generator(noise).cpu().detach().numpy()
                history['samples'].append(samples)

        return history

    def generate(self, num_samples, device='cpu'):
        noise = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.generator(noise).cpu().detach().numpy()
        return samples

    def load(self, path='saves', name='model'):
        self.generator.load_state_dict(torch.load(f'{path}/{name}_generator.pth'))
        self.discriminator.load_state_dict(torch.load(f'{path}/{name}_discriminator.pth'))
        history = np.load(f'{path}/{name}_history.npy', allow_pickle=True).item()
        return history

    def calculate_is(self, num_samples, device='cpu', feature='logits_unbiased', splits=10, **kwargs):
        noise = torch.randn(num_samples, self.latent_dim, device=device)
        fake_images = self.generator(noise)
        fake_images = (fake_images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)  # Scale to [0, 255]
        fake_images = fake_images.repeat(1, 3, 1, 1)  # Convert to RGB by repeating channels
        is_metric = InceptionScore(feature=feature, splits=splits, **kwargs).to(device)
        is_metric.update(fake_images)
        is_score = is_metric.compute()
        return is_score

    def calculate_fid(self, real_images, num_samples, device='cpu', feature=2048, reset_real_features=True,
                      normalize=True, **kwargs):
        noise = torch.randn(num_samples, self.latent_dim, device=device)
        fake_images = self.generator(noise)
        if normalize:
            fake_images = (fake_images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)  # Scale to [0, 255]
            real_images = (real_images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)  # Scale to [0, 255]
        fake_images = fake_images.repeat(1, 3, 1, 1)  # Convert to RGB by repeating channels
        real_images = real_images.repeat(1, 3, 1, 1)  # Convert to RGB by repeating channels
        fid_metric = FrechetInceptionDistance(feature=feature, reset_real_features=reset_real_features,
                                              normalize=normalize, **kwargs).to(device)
        fid_metric.update(real_images, real=True)
        fid_metric.update(fake_images, real=False)
        fid_score = fid_metric.compute()
        return fid_score.item()
