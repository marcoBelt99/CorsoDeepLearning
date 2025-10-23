'''
Addestrare una rete per un task di classificazione multi-label
Obiettivo: Classificazione multi-label su dataset MNIST custom fornito a lezione.
'''

# In questo tipo di classificazione, la rete mi deve dire, per ogni immagine, quante classi ci sono al suo interno.
# (quindi, non una sola classe ma multiclasse).


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

from train_and_validate import train_with_validation # Funzioni di allenamento e validazione
from dataset import MultiLabelMNIST
from model import MultiLabelNN

# Transformazioni per il dataset MNIST
transform = transforms.Compose([
    transforms.Resize((param.digit_size, param.digit_size)),
    transforms.ToTensor(),
])


# Carico MNIST
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Definisco il mio dataset
multi_label_mnist = MultiLabelMNIST(mnist_data, image_size=param.image_size, digit_size=param.digit_size,
                                    max_digits=param.max_digits)

# Lo divido in train e validation
train_size = int(0.8 * len(multi_label_mnist)) # voglio 80% training e il restante va nel dataset di validazione
val_size = len(multi_label_mnist) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(multi_label_mnist, [train_size, val_size])

# Ora ho i due dataset splittati e, passandoli al dataloader posso creare appunto il data loader che uso successivamente nella funzione
# di addestramento e di validazione del mio modello.

# Gestione dei data loader associati
train_loader = DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=param.batch_size, shuffle=False)

# Creazione del modello
model = MultiLabelNN(num_classi=param.num_classi) # istanzio la rete passandole il numero di classi, che è sempre 10, dato che sto lavorando su MNIST

# Definizione della funzione di perdita e dell'ottimizzatore
criterion = nn.BCELoss() # Binary Cross Entropy Loss per multi-label, anche perchè sto usando in uscita una sigmoide.
# Questo perchè in questo caso ho multi-label, e so che viene usata la stessa procedura che si usa per la classificazione binaria, ma con 
# più classi di uscita. In questo modo ho una classificazione binaria per ogni uscita della mia rete.

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Allenamento del modello con validazione usando l'opportuna funzione
train_with_validation(model, train_loader, val_loader, criterion, optimizer)
