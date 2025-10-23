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

# Questa funzione fa un giro di addestramento e una validazione alla fine
def train_with_validation(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train() # Imposto il modello in modalità addestramento
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()   # Azzeramento dei gradienti

            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Calcolo della perdita
            loss.backward() # Backward pass
            optimizer.step() # Aggiornamento dei pesi

            running_loss += loss.item()
            break
    print(f'Epoca [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # Validazione alla fine di ogni epoca
    validate(model, val_loader, criterion)


# Funzione di validazione
def validate(model, val_loader, criterion):
    model.eval() # Imposto il modello in modalità valutazione
    with torch.no_grad(): # Disabilito il calcolo dei gradienti
        for images, labels in val_loader: # validation set
            outputs = model(images) # Ottengo le predizioni
            predicted_labels = (outputs > 0.5).float() # Applico una soglia per ottenere le predizioni binarie

            # avendo dei numeri per ogni classe che va da 0 ad 1, mi sono definito tale policy (soglia), in cui dico che
            # se il valore supera 0.5, la rete ha individuato quella classe, altrimenti no.
            # Quindi, se l'output supera 0.5, allora me lo tieni, altrimenti me lo metti a 0.

            # Visualizzo alcune immagini con le etichette vere e quelle predette.
            # Con Matplotlib mi stampo una griglia con alcune delle immagini che ho usato per testare.
            # In 'Vere' trovo la label, mentre in 'Predette' ho il calcolo fatto dalla rete.
            # Man mano che addestro la rete, devo ottenere che la riga del 'Predette' deve diventare sempre di più
            # uguale alla riga 'Vere', col significato che la rete è stata in grado di trovare tutti i numeri che sono all'interno di quell'immagine.
            plt.figure(figsize=(12, 8))
            for i in range(6): # Mostro le prime 6 immagini
                plt.subplot(2, 3, i + 1)
                plt.imshow(images[i].squeeze(), cmap='gray')
                plt.title(f'Vere: {labels[i].numpy().astype(int)}\nPredette: {predicted_labels[i].numpy().astype(int)}')
                plt.axes('off')
            plt.show()
            break # Interrompo dopo il primo batch per visualizzare solo un campione
            
def validate_1(model, val_loader, criterion):
    model.eval() # Imposto il modello in modalità valutazione
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels) # Calcolo la perdita
            running_loss += loss.item()

            # Aggiungo le predizioni e le etichette per calcolare le metriche
            all_preds.append(outputs)
            all_labels.append(labels)
            break
    
    avg_loss = running_loss / len(val_loader)

    print(f'Validation Loss:  {avg_loss:.4f}')

# Funzione per testare il mdoello e visualizzare i risultati
# def test_and_visualize(model, val_loader)