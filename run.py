
#In case if torch summary is not installed, pip install torchsummary

# imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch import nn
from torchsummary import summary


from utilizations import show_tensor_images, weights_init, real_loss, fake_loss, Discriminator, Generator


# %%
#############################
# load MNIST
############################
train_augs = T.Compose(
    [
        T.RandomRotation((-20, 20)),
        T.ToTensor(),  # (h,w,c) -> (c,h,w), range [0, 1]
        T.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ]
)
trainset = datasets.MNIST('MNIST/', download=True, train=True, transform=train_augs)
print("The length of trainset is:", len(trainset))

image, label = trainset[100]
print(image.shape, label)
plt.imshow(image.squeeze(), cmap='gray')
plt.show()

# %%
################
# Hyperparameters
################


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 512
noice_dim = 64  # noise vector dimension to pass into generator

# optimizer params
lr = 2e-4
beta_1 = 0.85
beta_2 = 0.999  

# training variables
EPOCHS = 100

# %%
########
# Setup
########

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
print("Length of Trainloader =", len(trainloader))

# loading 1st batch and it's shape
dataiter = iter(trainloader)
images, labels = next(dataiter)
print("shapes of images and labels:", images.shape, labels.shape)

show_tensor_images(images)


D = Discriminator()
D.to(device)
D = D.apply(weights_init)
summary(D, input_size=(1, 28, 28))
D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta_1, beta_2))

G = Generator(noice_dim)
G.to(device)
G = G.apply(weights_init)
summary(G, input_size=(1, noice_dim))
G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta_1, beta_2))


# %%
###########
# Training
##########
for epoch in range(EPOCHS):
    total_d_loss = 0.0
    total_g_loss = 0.0
    pbar = tqdm(trainloader, desc=f"Train {epoch+1}/{EPOCHS}")

    for real_img, _ in pbar:
        real_img = real_img.to(device)
        current_batch_size = real_img.size(0)  # Handle last batch

        ######################
        # Train Discriminator
        ######################
        D_opt.zero_grad()

        # Train on real images
        D_pred_real = D(real_img)
        D_real_loss = real_loss(D_pred_real)

        # Train on fake images
        noice = torch.randn(current_batch_size, noice_dim, device=device)
        fake_img = G(noice)
        # FIXED: Detach fake images to prevent gradients flowing to Generator
        D_pred_fake = D(fake_img.detach())
        D_fake_loss = fake_loss(D_pred_fake)

        # Combine losses and update
        D_loss = (D_fake_loss + D_real_loss) / 2
        D_loss.backward()
        D_opt.step()

        total_d_loss += D_loss.item()

        ##################
        # Train Generator
        ##################
        G_opt.zero_grad()

        # FIXED: Generate fresh noise for generator training
        noice = torch.randn(current_batch_size, noice_dim, device=device)
        fake_img = G(noice)
        D_pred = D(fake_img)
        G_loss = real_loss(D_pred)  # Generator wants discriminator to think fakes are real

        G_loss.backward()
        G_opt.step()

        total_g_loss += G_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'D loss': f'{D_loss.item():.4f}',
            'G loss': f'{G_loss.item():.4f}'
        })

    avg_d_loss = total_d_loss / len(trainloader)
    avg_g_loss = total_g_loss / len(trainloader)

    print(f"Epoch: {epoch+1} | D loss: {avg_d_loss:.5f} | G loss: {avg_g_loss:.5f}")

    # Show generated images every epoch
    if (epoch + 1) % 5 == 0 or epoch == 0:
        show_tensor_images(fake_img)


# %%
# Now you can use Generator Network to generate handwritten images
print("\nGenerating final images...")
noise = torch.randn(batch_size, noice_dim, device=device)
generated_image = G(noise)
show_tensor_images(generated_image)


