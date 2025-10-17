'''
- Implementazione di una Rete Neurale Feedforward Semplice
- Obiettivo:  Implementa una rete neurale feedforward a due livelli e utilizza l'ottimizzatore SGD per 
  l'addestramento
    - Crea una rete neurale con un input layer di dimensione 2, un hidden layer di dimensione 4, e un
      output layer di dimensione 1
    - Addestra la rete su un set di dati sintetico (ad es. XOR)
    - Calcola e visualizza la funzione di perdita dopo ogni epoca
'''

import torch
import torch.nn as nn
import torch.optim as optim

## Definisco la Rete Neurale
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

## Creo il modello, la funzione di perdita e l'ottimizzatore
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

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