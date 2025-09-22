import torch
import torch.nn as nn
import torch.optim as optim # importo il modulo per usare gli ottimizzatori

## 1) Simulo alcuni dati di input e le etichette
inputs = torch.rand(10, 10) # creo 10 campioni, ciascuno di dimensione 10
labels = torch.randint(0, 2, (10,)) # queste sono 10 etichette per una classificazione binaria

## 2) Definisco un semplice modello sequenziale
model = nn.Sequential(
    nn.Linear(10, 50), # il mio strato di input ha 50 neuroni
    nn.ReLU(), # come funzione di attivazione utilizzo la ReLU
    nn.Linear(50, 2) # strato di output per due classi
)

## 3) Definisco l'ottimizzatore e lo Schedule del Learning Rate:
##    Utilizzo Adam come otimizzatore
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

'''
Cos'ho fatto?
Uno Scheduler riduce il learning rate di un fattore gamma pari a 0.9
ogni step_size (in questo caso con step_size=1, lo riduce ad ogni epoca)
'''

## 4) Creo la funzione di perdita
criterion = nn.CrossEntropyLoss()

## 5) Faccio un ciclo di addestramento
num_epochs = 10 # 10 epoche

for epoch in range(num_epochs):
    ## Forward pass
    outputs = model(inputs) # calcolo le predizioni del modello
    loss = criterion(outputs, labels) # calcolo la perdita (loss)
    
    ## Backward pass
    optimizer.zero_grad() # azzero i gradienti
    loss.backward() # eseguo il backward
    optimizer.step()
    # Aggiornamento del l.r. con lo scheduler:
    scheduler.step()

    # Stampo il l.r. corrente e la perdita
    print(f'Epoca [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Learning rate: {scheduler.get_last_lr()[0]}')