from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomDataloader(Dataset):
    def __init__(self, input_folder, target_folder, transform=None):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
    # Añade aquí más transformaciones si son necesarias
])

        # Asumimos que cada archivo en la carpeta de entrada tiene un correspondiente en la carpeta objetivo
        self.filenames = os.listdir(input_folder)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_folder, self.filenames[idx])
        target_path = os.path.join(self.target_folder, self.filenames[idx])

        input_image = Image.open(input_path).convert('L')  # Convertir a escala de grises si es necesario
        target_image = Image.open(target_path).convert('L')

        
        input_image = self.transform(input_image)
        target_image = self.transform(target_image)

        return input_image, target_image

# Transformaciones (si es necesario)
