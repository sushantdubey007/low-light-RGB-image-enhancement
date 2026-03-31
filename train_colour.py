import torch
from torch.utils.data import DataLoader
from dataset import LowLightDataset   # or your dataset class
from model import UNet

# 🔹 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# 🔹 Dataset paths
low_dir = "Train/low"
high_dir = "Train/high"

dataset = LowLightDataset(low_dir, high_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 🔹 Model
model = UNet().to(device)

# 🔹 Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# 🔹 Base Loss
l1_loss = torch.nn.L1Loss()


# 🔥 --- SSIM LOSS ---
import torch.nn.functional as F

def ssim_loss(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    sigma1 = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1**2
    sigma2 = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2**2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))

    return torch.clamp((1 - ssim_map) / 2, 0, 1).mean()


# 🔥 --- COLOR LOSS ---
def color_loss(output, target):
    r_out, g_out, b_out = output[:,0,:,:], output[:,1,:,:], output[:,2,:,:]
    r_t, g_t, b_t = target[:,0,:,:], target[:,1,:,:], target[:,2,:,:]

    loss = torch.mean((r_out - r_t)**2 + 
                      (g_out - g_t)**2 + 
                      (b_out - b_t)**2)
    return loss


# 🔹 Training
epochs = 20

for epoch in range(epochs):
    total_loss = 0

    for low, high in loader:
        low = low.to(device)
        high = high.to(device)

        # 🔹 Forward
        output = model(low)
        output = torch.clamp(output, 0, 1)

        # 🔹 Losses
        loss_l1 = l1_loss(output, high)
        loss_ssim = ssim_loss(output, high)
        loss_color = color_loss(output, high)

        # 🔥 FINAL LOSS (TUNABLE WEIGHTS)
        loss = loss_l1 + 0.1 * loss_ssim + 0.05 * loss_color

        # 🔹 Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# 🔹 Save model
torch.save(model.state_dict(), "unet_color_ssim.pth")
print("Model saved!")