import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_samples(samples, epoch):
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for i, sample in enumerate(samples):
        ax = axes[i // 4, i % 4]
        ax.imshow(sample[0], cmap='gray')
        ax.axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.show()


def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['G_loss'], label='Generator Loss')
    plt.plot(history['D_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def save(generator, discriminator, history, teacher_models=None, path='saves', name='model'):
    torch.save(generator.state_dict(), f'{path}/{name}_generator.pth')
    torch.save(discriminator.state_dict(), f'{path}/{name}_discriminator.pth')
    if teacher_models is not None:
        for i, teacher_model in enumerate(teacher_models):
            torch.save(teacher_model.state_dict(), f'{path}/{name}_teacher_{i}.pth')
    np.save(f'{path}/{name}_history.npy', history)


def load(generator, discriminator, teacher_models=None, path='saves', name='model'):
    generator.load_state_dict(torch.load(f'{path}/{name}_generator.pth'))
    discriminator.load_state_dict(torch.load(f'{path}/{name}_discriminator.pth'))
    if teacher_models is not None:
        for i, teacher_model in enumerate(teacher_models):
            teacher_model.load_state_dict(torch.load(f'{path}/{name}_teacher_{i}.pth'))
    history = np.load(f'{path}/{name}_history.npy', allow_pickle=True).item()
    return generator, discriminator, history
