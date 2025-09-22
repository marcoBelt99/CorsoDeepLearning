import torch
from torch.utils.data import DataLoader, WeightedRandomSampler # Importo il dataloader e il sampler
from torchvision import datasets, transforms # importo datasets e transforms

# Faccio in modo che le etichette abbiano il significato di 0=mela, 1=banana, 2=arancia
labels = torch.tensor( [0, 1, 2, 0, 0, 1, 0, 2, 2, 2, 1, 0] )

# Calcolo la frequenza di ogni classe, e la stampo
class_count = torch.bincount( labels )
print("Frequenza delle classi: ", class_count)

# Calcolo i pesi, che sono inversamente proporzionali alle frequenze
weights = 1. / class_count.float()
print("Pesi inversi: ", weights)

# Assegno ad ogni elemento nel DS il peso corrispondente alla sua classe
sample_weights = weights[labels]
print("Pesi dei campioni: ", sample_weights)

# Creo un'istanza del WeightedRandomSampler
sampler = WeightedRandomSampler( weights=sample_weights, num_samples=len(sample_weights), replacement=True )

## Poi, carico il Dataset:
# Uso transofrms.Compose
transform = transforms.Compose( [transforms.ToTensor()] )

# Come dataset decido di usare Fake data, quindi: size 100, image_size=3x224x224, num_classes=3
dataset = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=3, transform=transform)

## Creo il DataLoader usando il sampler che ho definito prima
data_loader = DataLoader(dataset, batch_size=4, sampler=sampler)

# Itero sul DL e stampo le etichette dei batch per vedere il risultato
for images, labels in data_loader:
    # Stampo il risultato
    print("Etichette del batch: ", labels)


'''
Questo esempio è stato utile per capire il DataLoader di PyTorch, un componente importante per la gestione efficiente dei dati
nel Deep Learning. Con queste funzionalità posso ottimizzare il processo di addestramento, e sfruttare al meglio le risorse
computazionali, quindi migliorare le prestazioni dei miei futuri modelli.
'''
