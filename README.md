Low-Light Image Enhancement using U-Net

This project implements a U-Net based deep learning model to enhance low-light RGB images.
The model learns a mapping from dark images → well-lit images using paired training data.

Features
* U-Net architecture for image-to-image enhancement
* Works with RGB images (3-channel input/output)
* Training on paired dataset (low / high images)
* Supports GPU (CUDA) and CPU
* Easy testing on custom images
* Visualization of results

Files details 
- dataset.py # Custom dataset loader 
- model.py # U-Net architecture 
- train.py # Training script 
- test.py # Inference / testing script 
- unet_model.pth # Trained model weights 
- test_low.png # Sample low-light image 
- enhanced.png # Output enhanced image

Model Overview
* Input: Low-light RGB image (3, H, W)
* Output: Enhanced RGB image (3, H, W)
* Architecture: Encoder–Decoder (U-Net)
* Skip connections preserve spatial details

Install dependencies
pip install torch torchvision matplotlib opencv-python pillow tqdm

Training
Update dataset paths in train.py:

low_dir = "dataset/low"
high_dir = "dataset/high"

Run training:
python train.py

Output:
Trained model saved as unet_model.pth


Testing / Inference
Place your test image:
test_low.png

Run:
python test.py

Output:
Enhanced image saved as:
enhanced.png
Visualization window showing:
Input (Low Light)
Enhanced Output
