import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import UNet   # make sure model.py is in same folder


def main():
    # 🔹 Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 🔹 Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully")

    # 🔹 Transform (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])

    # 🔹 Load test image
    img_path = "test_low.png"   # <-- change if needed
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 🔹 Inference
    with torch.no_grad():
        output = model(input_tensor)

    # 🔹 Convert output
    output_img = output.squeeze().cpu().permute(1, 2, 0).numpy()
    output_img = np.clip(output_img, 0, 1)   # IMPORTANT

    # 🔹 Display
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Input (Low Light)")
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title("Enhanced Output")
    plt.imshow(output_img)

    plt.show()

    # 🔹 Save output
    output_img = (output_img * 255).astype(np.uint8)
    cv2.imwrite("enhanced.png", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

    print("Saved as enhanced.png")


if __name__ == "__main__":
    main()