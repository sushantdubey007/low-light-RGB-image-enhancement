import torch
from torch.utils.data import DataLoader
from dataset import LowLightDataset
from model import UNet
import matplotlib.pyplot as plt

# 🔹 Paths
low_dir = "Train/low"
high_dir = "Train/high"

# 🔹 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# 🔹 Dataset
dataset = LowLightDataset(low_dir, high_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 🔹 Model
model = UNet().to(device)

# 🔹 Loss + Optimizer
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 🔹 Training
epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for low, high in loader:
        low = low.to(device)
        high = high.to(device)

        output = model(low)
        loss = criterion(output, high)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

# 🔹 Save model
torch.save(model.state_dict(), "unet_model.pth")