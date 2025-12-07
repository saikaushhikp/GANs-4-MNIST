
#In case if torch summary is not installed, pip install torchsummary

# imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torch import nn
from torchsummary import summary
import os


from utilizations import show_tensor_images, weights_init, discriminator_loss, generator_loss, Discriminator, Generator


# %%
################
# Hyperparameters
################


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# hyperparams tuned for stability
batch_size = 128
noice_dim = 100
lr = 2e-4
beta1 = 0.5   # DCGAN recommendation
beta2 = 0.999
EPOCHS = 100 


# %%
#############################
# load MNIST
############################
# data augmentation / normalization
train_augs = T.Compose([
    T.RandomRotation((-10, 10)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))  # map to [-1,1]
])


trainset = datasets.MNIST('MNIST/', download=True, train=True, transform=train_augs)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

print("The length of trainset is:", len(trainset))

image, label = trainset[100]
print(image.shape, label)
plt.imshow(image.squeeze(), cmap='gray')
plt.show()

# %%
########
# Setup
########

# loading 1st batch and it's shape
dataiter = iter(trainloader)
images, labels = next(dataiter)
print("shapes of images and labels:", images.shape, labels.shape)

show_tensor_images(images)


D = Discriminator().to(device)
G = Generator(noice_dim).to(device)

D.apply(weights_init)
G.apply(weights_init)

D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

# Optional: LR schedulers can help, but not required initially

# Diagnostics
print("Trainloader length:", len(trainloader))
summary(D, input_size=(1,28,28))
summary(G, input_size=(1, noice_dim))


# %%
###########
# Training
##########

fixed_noise = torch.randn(64, noice_dim, device=device)  # fixed for progress visualization
d_losses = []
g_losses = []

os.makedirs("gan_outputs", exist_ok=True)


for epoch in range(1, EPOCHS+1):
    D.train()
    G.train()
    running_d = 0.0
    running_g = 0.0
    pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

    for real_imgs, _ in pbar:
        real_imgs = real_imgs.to(device)
        b_size = real_imgs.size(0)

        # ----------------------
        # Train Discriminator
        # ----------------------
        D_opt.zero_grad()
        # Real logits
        logits_real = D(real_imgs)
        # Fake images (detach so gradients not propagated into G)
        noise = torch.randn(b_size, noice_dim, device=device)
        fake_imgs = G(noise).detach()
        logits_fake = D(fake_imgs)

        d_loss = discriminator_loss(logits_real, logits_fake, smoothing=0.9, noisy_labels=0.05)
        d_loss.backward()
        D_opt.step()

        # ----------------------
        # Train Generator
        # ----------------------
        G_opt.zero_grad()
        noise = torch.randn(b_size, noice_dim, device=device)  # fresh noise
        fake_imgs = G(noise)
        logits_fake_for_g = D(fake_imgs)   # don't detach
        g_loss = generator_loss(logits_fake_for_g)
        g_loss.backward()
        G_opt.step()

        running_d += d_loss.item()
        running_g += g_loss.item()

        pbar.set_postfix({'D_loss': f'{d_loss.item():.4f}', 'G_loss': f'{g_loss.item():.4f}'})

    avg_d = running_d / len(trainloader)
    avg_g = running_g / len(trainloader)
    d_losses.append(avg_d)
    g_losses.append(avg_g)

    print(f"Epoch {epoch:02d} | D_loss: {avg_d:.4f} | G_loss: {avg_g:.4f}")

    # Save samples from fixed noise to monitor progress
    G.eval()
    with torch.no_grad():
        samples = G(fixed_noise)
        save_image((samples * 0.5 + 0.5), f"gan_outputs/fixed_noise_epoch_{epoch:03d}.png", nrow=8)
    G.train()


# %%

plt.figure(figsize=(8,4))
plt.plot(d_losses, label='D loss')
plt.plot(g_losses, label='G loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training losses')
plt.grid(True)
plt.show()

# Show last sample grid
with torch.no_grad():
    final_noise = torch.randn(64, noice_dim, device=device)
    final_samples = G(final_noise)
    show_tensor_images(final_samples)

# Save models
torch.save(G.state_dict(), "gan_outputs/generator_final.pth")
torch.save(D.state_dict(), "gan_outputs/discriminator_final.pth")

print("Training finished. Models and sample images saved to ./gan_outputs")


