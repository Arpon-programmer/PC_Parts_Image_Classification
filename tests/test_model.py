import torch
from test_data_preparation import process_images

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(256 * 256 * 1, 14)
)

test_data_dir = "/Users/arponbiswas/Computer-Vision-Projects/Image_classification_projects/PC_Parts_Image_Classification/Data/test_images"
pc_parts_classes = ['cables', 'case', 'cpu', 'gpu', 'hdd', 'headset', 'keyboard', 'microphone', 'monitor', 'motherboard', 'mouse', 'ram', 'speakers', 'webcam']

def model_operation(test_data_dir, image_channel):
    try:
        processed_images, image_names = process_images(test_data_dir, image_channel)
    except Exception as e:
        print(f"Error processing images: {e}")
        return [], []
    pred_images = []
    for idx, processed_image in enumerate(processed_images):
        try:
            output = model(processed_image)
            pred = torch.argmax(output, dim=1)
            pred_images.append(pc_parts_classes[pred])
        except Exception as e:
            print(f"Error predicting image '{image_names[idx]}': {e}")
            pred_images.append("Error")
    return pred_images, image_names

if __name__ == "__main__":
    try:
        outputs, image_names = model_operation(test_data_dir, 1)
        for name, output in zip(image_names, outputs):
            print(f"Image: {name}, Predicted Class: {output}")
    except Exception as e:
        print(f"Unexpected error: {e}")
