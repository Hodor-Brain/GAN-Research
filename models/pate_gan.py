import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


class PATE_GAN:
    def __init__(
            self,
            generator,
            teacher_models,
            student_discriminator,
            latent_dim,
    ):
        self.generator = generator
        self.teacher_models = teacher_models
        self.student_discriminator = student_discriminator
        self.latent_dim = latent_dim
        self.num_teachers = len(teacher_models)

    def train_teacher_models(
            self,
            dataset,
            teacher_optimizers,
            criterion=nn.BCELoss(),
            epochs=20,
            batch_size=64,
            device='cpu',
            verbose=True
    ):
        # Splitting the dataset into subsets for each teacher
        subsets = torch.utils.data.random_split(dataset, [len(dataset) // self.num_teachers] * self.num_teachers)
        teacher_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in
                           subsets]

        # Training each teacher model independently
        for teacher_id, (model, optimizer, loader) in enumerate(
                zip(self.teacher_models, teacher_optimizers, teacher_loaders)):
            if verbose:
                pbar = tqdm(total=len(loader) * epochs, desc=f'Teacher {teacher_id + 1}', unit='batch', leave=True)
            for epoch in range(epochs):
                for real_data, _ in loader:
                    real_data = real_data.to(device)
                    real_labels = torch.ones(real_data.size(0), 1, device=device)
                    optimizer.zero_grad()
                    output = model(real_data)
                    loss = criterion(output, real_labels)
                    loss.backward()
                    optimizer.step()
                    if verbose:
                        pbar.update(1)
                if verbose:
                    pbar.set_description(f'Teacher {teacher_id + 1} Epoch {epoch + 1}/{epochs}')
            if verbose:
                pbar.close()

    def _aggregate_teacher_predictions(self, data, noise_scale, device):
        with torch.no_grad():
            votes = np.zeros((data.size(0), 1))
            for model in self.teacher_models:
                output = model(data).cpu().numpy()
                votes += (output > 0.5).astype(int)
            noise = np.random.laplace(loc=0.0, scale=noise_scale, size=votes.shape)
            noisy_votes = votes + noise
            aggregated_predictions = (noisy_votes > (self.num_teachers / 2)).astype(int)
        return torch.tensor(aggregated_predictions, dtype=torch.float32).to(device)

    def train(
            self,
            dataloader,
            optimizer_G,
            optimizer_S,
            criterion=nn.BCELoss(),
            epochs=20,
            noise_scale=1.0,
            device='cpu'
    ):
        history = {'G_loss': [], 'D_loss': [], 'samples': []}

        for epoch in range(epochs):
            G_loss_total = 0.0
            S_loss_total = 0.0

            for real_data, _ in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
                batch_size = real_data.size(0)
                real_data = real_data.to(device)
                real_labels = self._aggregate_teacher_predictions(real_data, noise_scale, device)
                fake_labels = torch.zeros(batch_size, 1, device=device)

                # Train Student Discriminator
                optimizer_S.zero_grad()
                noise = torch.randn(batch_size, self.latent_dim, device=device)
                fake_data = self.generator(noise)
                real_output = self.student_discriminator(real_data)
                fake_output = self.student_discriminator(fake_data.detach())
                real_loss = criterion(real_output, real_labels)
                fake_loss = criterion(fake_output, fake_labels)
                S_loss = real_loss + fake_loss
                S_loss.backward()
                optimizer_S.step()

                # Train Generator
                optimizer_G.zero_grad()
                noise = torch.randn(batch_size, self.latent_dim, device=device)
                fake_data = self.generator(noise)
                output = self.student_discriminator(fake_data)
                G_loss = criterion(output, real_labels)
                G_loss.backward()
                optimizer_G.step()

                G_loss_total += G_loss.item()
                S_loss_total += S_loss.item()

            avg_G_loss = G_loss_total / len(dataloader)
            avg_S_loss = S_loss_total / len(dataloader)
            history['G_loss'].append(avg_G_loss)
            history['D_loss'].append(avg_S_loss)

            print(
                f"Epoch [{epoch + 1}/{epochs}], Generator Loss: {avg_G_loss}, Student Discriminator Loss: {avg_S_loss}")

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
        self.student_discriminator.load_state_dict(torch.load(f'{path}/{name}_discriminator.pth'))
        for i in range(self.num_teachers):
            self.teacher_models[i].load_state_dict(torch.load(f'{path}/{name}_teacher_{i}.pth'))
        history = np.load(f'{path}/{name}_history.npy', allow_pickle=True).item()
        return history

    def calculate_is(self, num_samples, device='cpu', feature='logits_unbiased', splits=10, **kwargs):
        noise = torch.randn(num_samples, self.latent_dim, device=device)
        fake_images = self.generator(noise)
        fake_images = (fake_images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        fake_images = fake_images.repeat(1, 3, 1, 1)
        is_metric = InceptionScore(feature=feature, splits=splits, **kwargs).to(device)
        is_metric.update(fake_images)
        is_score = is_metric.compute()
        return is_score

    def calculate_fid(self, real_images, num_samples, device='cpu', feature=2048, reset_real_features=True,
                      normalize=True, **kwargs):
        noise = torch.randn(num_samples, self.latent_dim, device=device)
        fake_images = self.generator(noise)
        if normalize:
            fake_images = (fake_images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
            real_images = (real_images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        fake_images = fake_images.repeat(1, 3, 1, 1)
        real_images = real_images.repeat(1, 3, 1, 1)
        fid_metric = FrechetInceptionDistance(feature=feature, reset_real_features=reset_real_features,
                                              normalize=normalize, **kwargs).to(device)
        fid_metric.update(real_images, real=True)
        fid_metric.update(fake_images, real=False)
        fid_score = fid_metric.compute()
        return fid_score.item()
