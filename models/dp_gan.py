import torch
import torch.nn as nn
from tqdm import tqdm


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
