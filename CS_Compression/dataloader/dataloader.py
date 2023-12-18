from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np

from torchvision import transforms

class DatasetLoader(Dataset):
    def __init__(self, image_paths, augment=True):
        self.image_paths = [os.path.abspath(os.path.join(image_paths, archivo))
                            for archivo in os.listdir(image_paths)
                            if os.path.isfile(os.path.join(image_paths, archivo))]

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Volteo horizontal aleatorio
                transforms.RandomRotation(15),      # Rotación aleatoria en un rango de ±15 grados
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Transformación afín aleatoria
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')  # Asegúrate de que la imagen se carga como blanco y negro

        if self.transform:
            image = self.transform(image)
        return image
