import torch
import torch.nn as nn 

# nn è il sottomodulo specifico di PyTorch
# per la costruzione di RN, che ospita vari componenti per creare
# strati nei modelli di RN.

# 2) Simulo un dataset di 100 campioni, ciascuno di dimensione
# 784, Es) potrebbero essere immagini 28x28 in bianco e nero

# Procedo allora a creare il primo tensore

inputs = torch.randn(100, 784)

# 3) Creo le labels, quindi creo 100 etichette (1 per ogni campione)
# per un eventuale problema di classificazione a 10 classi
labels = torch.randint(0, 10, (100,))

# 4) Definisco il modello sequenziale (gli input vanno da sx verso dx)
model = nn.Sequential(
    # Inizio creando uno strato completamente connesso con 256 neuroni
    nn.Linear(in_features=784, out_features=256) , # inizio con 784 features di input che mappo su 256 features sul 1° layer nascosto
    # Inserisco la funzione di attivazione
    nn.ReLU(), # la ReLu introduce la non linearità, ma permette di apprendere relazioni complesse nei dati
    # Metto poi lo strato di output: in input prendo l'output dello strato precedente
    nn.Linear(in_features=256, out_features=10) # mappo le 256 features a 10 features di output. Che appunto
    # queste 10 potrebbero essere ad es. le probabilità di 10 classi diverse in un problema eventuale di classificazione.
)

'''
Quindi, fin qui:
il modello inizia con uno strato Lineare, che trasforma l'input
di 784 features (che potrebbero essere i pixel di un'immagine 28x28
in bianco e nero) in un vettore di 256 caratteristiche.
Poi viene introdotta la funzione di attivazione ReLU per introdurre
la non linearità: essa cioè consente di apprendere relazioni complesse
nei dati.
Tale funzione di attivazione è seguita da un'altro strato lineare, che mappa
le 256 features a 10 features di output (che potrebbero essere ad esempio le
probabilità di 10 classi diverse in un eventuale problema di classificazione).

Il modello sequenziale si usa per chi è agli inizi nel DL, per prototipare.
La facilità è a discapito delle flessibilità: ci sono approcci più complessi
che permettono architetture di rete con connessioni più elaborate. Infatti,
si tende in questi casi ad usare altri sottoclassi di nn, che offrono maggior
flessibilità. 
'''

# 5) Addestro il modello: ciò richiede di dover definire la funzione di perdita
# e di dover poi scegliere un ottimizzatore.

# 6) Funzione di perdita: uso la Cross-Entropy Loss, perchè ho un problema
# di classificazione
criterion = nn.CrossEntropyLoss()

# 7) Definisco l'ottimizzatore, dando un valore al learning rate.
# Come ottimizzatore uso Adam, perchè è semplice da usare ed efficace.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 8) Creo il numero di cicli di addestramento
num_epochs = 10

'''
L'addestramento del modello avviene in epoche, ovvero in cicli.
Ogni ciclo è composto da:
- forward pass: in cui faccio passare i dati attraverso il modello
- backward pass: ottimizzo i pesi del modello in base all'errore calcolato
'''

# 9) Creo il ciclo di addestramento
for epoch in range(num_epochs):
    # 10) Creo il forward pass, calcolando le predizioni del modello e la perdita
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    # 11) Faccio il backward pass: azzero i gradienti, eseguo il backward e aggiorno i pesi
    optimizer.zero_grad()
    '''
    Se non chiamo zero_grad(), i gradienti del ciclo di addestramento precedente
    verrebbero sommati ai gradienti del ciclo attuale. Questo porterebbe ad un aggiornamento non corretto
    dei parametri. Quindi, è fondamentale chiamare optimizer.zero_grad() prima di ogni nuova iterazione di 
    backpropagation, per garantire che il calcolo dei gradienti parti da 0, e quindi rappresenti esclusivamente
    le variazioni relative all'epoca corrente.
    '''
    loss.backward()
    optimizer.step()

    # 12) Stampa dei risultati
    print(f"Epoca [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")