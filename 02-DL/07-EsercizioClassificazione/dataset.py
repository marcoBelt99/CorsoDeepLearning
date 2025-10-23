import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import parameters as param # Parametri


class MultiLabelMNIST(Dataset):
    def __init__(self, mnist_data, image_size=64, digit_size=64, max_digits=3):
        self.mnist_data = mnist_data
        self.image_size = image_size
        self.digit_size = digit_size
        self.max_digits = max_digits

        # Genera tutte le possibili posizioni non sovrapposte in una griglia
        self.positions = [
            (x, y)
            for x in range(0, self.image_size - self.digit_size + 1, self.digit_size)
            for y in range(0, self.image_size - self.digit_size + 1, self.digit_size)
        ]

    def __len__(self):
        return len(self.mnist_data)
    
    def __getitem__(self, idx):
        # Numero casuale di cifre da posizionare
        num_digits = random.randint(1, self.max_digits)

        # Creare un tensore vuoto per l'immagine 64x64
        combined_image = torch.zeros(1, self.image_size, self.image_size)
        labels = torch.zeros(param.num_classi)

        # Seleziona un sottoinsieme casuale di posizioni dalla lista pre-generata
        selected_positions = random.sample(self.positions, num_digits)

        # Selezionare cifre casuali e posizionarle nelle posizioni scelte
        for i in range(num_digits):
            rand_idx = random.randint(0, len(self.mnist_data) -1 )
            img, label = self.mnist_data[rand_idx]
            labels[label] = 1

            # Posiziono la cifra nella posizione scelta
            x_offset, y_offset = selected_positions[i]
            combined_image[:, y_offset:y_offset + self.digit_size, x_offset:x_offset + self.digit_size] = img

        return combined_image, labels
        