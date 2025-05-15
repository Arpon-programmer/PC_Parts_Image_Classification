from torchvision import transforms

train_trans = transforms.Compose([
    # 1. Geometric Transformations (with resizing)
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flips for left-right consistency
    transforms.RandomRotation(10),  # Minor rotation for realistic variation
    transforms.RandomAffine(
        degrees=0,  # No additional rotation
        translate=(0.05, 0.05),  # Small positional variance
        scale=(0.9, 1.1)  # Conservative scaling for proportion preservation
    ),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3, interpolation=3),  # Subtle 3D perspective

    # 2. Color Augmentations
    transforms.ColorJitter(
        brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
    ),  # Mild color variation for natural lighting

    # 3. Conversion and Normalization
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5597623586654663, 0.5584744215011597, 0.5615870952606201],
        std=[0.2830338776111603, 0.27588963508605957, 0.2759666442871094]
    )
])

val_trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5597623586654663, 0.5584744215011597, 0.5615870952606201],
        std=[0.2830338776111603, 0.27588963508605957, 0.2759666442871094]
    )
])

def train_transform(image):
    return train_trans(image)

def val_transform(image):
    return val_trans(image)