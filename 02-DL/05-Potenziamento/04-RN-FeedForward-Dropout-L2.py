'''
- Regolarizzazione L2 e Dropout
- Obiettivo:  Aggiungi regolarizzazione L2 e Dropout alla rete neurale del precedente esercizio e osserva l'effetto sulla funzione di perdita
    - Aggiungi la regolarizzazione L2 alla loss
    - Implementa il Dropout sul hidden layer con una probabilità di dropout del 50%
    - Confronta la funzione di perdita con e senza regolarizzazione/Dropout
'''

import torch
import torch.nn as nn
import torch.optim as optim

## Definisco la Rete Neurale con Droput e L2
class RegularizedNN(nn.Module):
    def __init__(self):
        super(RegularizedNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.dropout = nn.Dropout(p=0.5) # lo è già di default 0.5
        self.fc2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) # aggiunto il dropout
        x = self.fc2(x)
        return x

## Creo il modello, la funzione di perdita con L2 e l'ottimizzatore
model = RegularizedNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01) # regolarizzazione L2

## Creo un dataset sintetico (XOR)
inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
etichette = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

## Addestro il modello
for epoch in range(1000):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, etichette)

    # Backward pass e ottimizzazione
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoca {epoch}, Perdita: {loss.item()}')

## Test del modello
with torch.no_grad():
    test_outputs = model(inputs)
    print('Output del test:\n', test_outputs)



# Pare che senza Dropout ma con L2 vada bene, e meglio della solzione 03-RN-FeedForward.py