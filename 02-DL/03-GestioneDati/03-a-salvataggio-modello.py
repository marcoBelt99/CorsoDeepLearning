import os
import torch
import torch.nn as nn

# Percorso della directory in cui si trova questo script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

# Richiamo un modello sequenziale con: uno strato di input Linear, funzione di attivazione ReLU() e poi un altro strato come output:
model = torch.nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

## Salvataggio dello stato del modello

torch.save( model.state_dict(), MODEL_PATH )
print(f"Modello salvato con successo in: {MODEL_PATH}")


'''
La destinazione del salvataggio è un file model.pth.
Salvare solo il state_dict() è più leggero e flessibile rispetto al salvataggio dell'intero
modello, dato che mi consente di modificare la struttura del modello se necessario, e poi
ricaricare solo i parametri!
'''