import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from model import UNet


def main():
    # 🔹 Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 🔹 Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_color_ssim.pth", map_location=device))
    model.eval()
    print("Model loaded successfully")

    # 🔹 Transform (NO resize here)
    transform = transforms.ToTensor()

    # 🔹 Load test image
    img_path = "test_low.png"
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 🔥 Save original size
    orig_h, orig_w = img.shape[:2]

    # Convert to tensor (original size)
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 🔹 Inference
    with torch.no_grad():
        output = model(input_tensor)

    # 🔥 Ensure output same size (safety)
    output = F.interpolate(
        output,
        size=(orig_h, orig_w),
        mode='bilinear',
        align_corners=False
    )

    # 🔹 Convert output
    output_img = output.squeeze().cpu().permute(1, 2, 0).numpy()
    output_img = np.clip(output_img, 0, 1)

    # 🔹 Display
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title(f"Input ({orig_w}x{orig_h})")
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title(f"Output ({orig_w}x{orig_h})")
    plt.imshow(output_img)

    plt.show()

    # 🔹 Save output
    output_img = (output_img * 255).astype(np.uint8)
    cv2.imwrite("enhanced.png", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

    print("Saved as enhanced.png")


if __name__ == "__main__":
    main()