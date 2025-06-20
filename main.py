import torch
import torchvision.transforms as transforms
from PIL import Image

#model = torch.jit.load('Path of the model', map_location='cuda' if torch.cuda.is_available() else 'cpu')

pc_parts_classes = ['cables','case','cpu','gpu','hdd','headset','keyboard',
 'microphone',
 'monitor',
 'motherboard',
 'mouse',
 'ram',
 'speakers',
 'webcam']

def preprocess_image(image_path, img_type="RGB"):
    if img_type.lower() == "rgb":
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

def operation(img_path):
    processed_img = preprocess_image(img_path)
    out = model(processed_img)
    pred = torch.argmax(out, dim=1)
    pred_img = pc_parts_classes(pred)
    return pred_img

if __name__ == "__main__":
    while True:
        img_path = input('Give the image path : ')
        prediction = operation(img_path)
        print(f"Predicted image {prediction}.")