import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from vit_pytorch import ViT

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the probabilistic ViT model
class ProbabilisticViT(nn.Module):
    def __init__(self, num_classes, latent_dim=128):
        super(ProbabilisticViT, self).__init__()
        self.vit = ViT()
        self.fc1_mu = nn.Linear(in_features=self.vit.embed_dim, out_features=latent_dim)
        self.fc1_logvar = nn.Linear(in_features=self.vit.embed_dim, out_features=latent_dim)
        self.fc2_mu = nn.Linear(in_features=latent_dim, out_features=num_classes)
        self.fc2_logvar = nn.Linear(in_features=latent_dim, out_features=num_classes)

    def forward(self, x):
        out = self.vit(x)
        z_mu = self.fc1_mu(out)
        z_logvar = self.fc1_logvar(out)
        z = self.reparameterize(z_mu, z_logvar)
        y_mu = self.fc2_mu(z)
        y_logvar = self.fc2_logvar(z)
        return y_mu, y_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# Define the loss function
def variational_loss(y_true, y_mu, y_logvar):
    reconstruction_loss = F.cross_entropy(y_mu, y_true)
    kl_divergence = -0.5 * torch.sum(1 + y_logvar - y_logvar.exp() - y_mu.pow(2))
    return reconstruction_loss + kl_divergence

# Set the hyperparameters
num_classes = 10
latent_dim = 128
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the model instance
model = ProbabilisticViT(num_classes, latent_dim).to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        y_mu, y_logvar = model(images)
        loss = variational_loss(labels, y_mu, y_logvar)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'vit_b32_probabilistic.pth')

print("Training finished.")
