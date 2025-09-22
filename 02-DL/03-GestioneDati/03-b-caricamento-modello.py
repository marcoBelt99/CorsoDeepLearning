import os
import torch
import torch.nn as nn

# Percorso della directory in cui si trova questo script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

'''
Per fare il caricamento, devo caricare lo stato del modello in un nuovo modello.
'''



## Quindi, ricreo un modello del tutto simile a quello precedente:
model_loaded = torch.nn.Sequential(
    nn.Linear(784, 128), # strato di input
    nn.ReLU(), # funzione di attivazione
    nn.Linear(128, 10) # strato di output
)

## Poi vado a fare il caricamento
model_loaded.load_state_dict( torch.load(MODEL_PATH) )
print(f"Modello caricato con successo da: {MODEL_PATH}")


'''
Attenzione: devo sempre assicurarmi che l'architettura del modello in cui carico i pesi
sia identica a quella del modello da cui sono stati salvati i pesi.
'''