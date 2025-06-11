import torch
import torchvision.transforms as transforms
from PIL import Image

model = torch.jit.load('Path of the model', map_location='cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_image(image_path, img_type="RGB"):
    if img_type.loawer() == "rgb":
        image = Image.open(image_path).convert("RGB")
    elif img_type.lower() == "grayscale":
        image = Image.open(image_path).convert("L")
    else:
        raise ValueError("Unsupported image type. Use 'RGB' or 'grayscale'.")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)