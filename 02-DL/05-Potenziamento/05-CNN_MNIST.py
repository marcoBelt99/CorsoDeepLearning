'''
- Implementazione di una Rete Convoluzionale (CNN)
    - Obiettivo:  Implementa una semplice rete convoluzionale e addestrala su un dataset come MNIST
    - Crea una rete convoluzionale con due layer convoluzionali seguiti da un layer fully connected
    - Addestralo su un dataset di benchmark (cifar10, mnist, …)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

## Definire la rete neurale convoluzionale
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5) #definisco che in output avrò 16 canali
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # maxpooling 2x2 mi dimezza l'immagine
        self.softmax = nn.Softmax(dim=1) # softmax perchè mi serve per fare la classificazione a 10 classi

    def forward(self, x):
        ## Qui ho due blocchi convoluzionali che sono composti da: convoluzione, attivazione e pooling.
        # Passo l'input x attraverso il primo strato convoluzionale e applica ReLU
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x) # Applica il pooling

        # Passo l'input x attraverso il secondo strato convoluzionale e applica ReLU
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) # Applica il pooling

        # Appiattisco il tensore (3-D) per fare in modo che questo venga "mangiato" dai layer fully connected.
        # Ovviamente devo fare questa operazione prima di entrare nel layer fully-connected  
        x = x.view(-1, 32 * 4 * 4 ) # reshape di un tensore in base alle dimensioni che voglio io: l'importante è che il numero complessivo di numero all'interno
        # del tensore sia sempre quello.
        
        # Mettendo -1 gli sto dicendo di calcolare tu il numero di dati che rimangono.
        # N.B: Il risultato di 32 * 4 * 4 = 512 è proprio la stessa dimensione che prende in input il primo strato fully connected
        # Dopo questa operazione il tensore è diventato un vettore

        # Passo attraverso il primo strato fully connected e applico ReLU
        x = self.fc1(x)
        x = self.relu(x)

        # (arrivato qui, il valore 512 sarà diventato 128, ed è pronto per passare in input al secondo strato fully connected)

        # Passo attraverso il secondo strato fully connected
        x = self.fc2(x)

        # (arrivato qui, da 128 passo a 10, pronto per entrare nel layer di output)

        # Applico la softmax per ottenere le probabilità
        x = self.softmax(x)

        return x

## Imposto il dispositivo: (il computer che sto usando ha o meno la GPU?)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Carico il dataset: in questo caso è MNIST

# Definisco una composizione di trasformazioni, che mi servono per dire al mio DataLoader che,
# quando mi dovrà restituire un batch di dati me lo deve trasformare in:
transform = transforms.Compose([
    transforms.ToTensor(), # tensore, in modo che io possa lavorarci tranquillamente con PyTorch
    transforms.Normalize((0.5), (0.5)) # e poi lo deve normalizzare, secondo la media e la deviazione standard. Questi valori dipendono dalla situazione
                            # da che rete sto usando, se sono state addestrate su un altro DS che hanno una media e deviazione standard diverse
                            # (dipende un po' dalla situazione in cui mi trovo). Quindi, ho la possibilità di normalizzare i dati in input alla rete direttamente qua
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) # train=True
# mi scarica il dataset nella cartella che gli dico io './data' in questo caso.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# il batch_size lo decido io (in genere è 64 o 32). Per il train shuffle=True, in modo da avere una situazione
# più randomica e realistica

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) # train=False
# mi scarica il dataset nella cartella che gli dico io './data' in questo caso.
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
# il batch_size lo decido io (in genere è 64 o 32). Per il test shuffle=False

## Se metto un brakpoint esattamente qui ed eseguio a console di debug len(trainloader) e poi len(testloader),
## vedo quanti batch hanno rispettivamente il dataset di training e quello di test

## Creo il modello, la funzione di perdita e l'ottimizzatore
model = SimpleCNN().to(device) #.cuda() # cerco di spostare il grafo della rete su GPU, in modo che sia già pronto per il calcolo
criterion = nn.CrossEntropyLoss() # dato che è una classificazione multiclasse
optimizer = optim.Adam(model.parameters()) # come so, all'ottimizzatore passo i parametri del modello che mi deve andare ad aggiornare
# se non gli passo altro, mi prende il learning rate di default

## Addestro il modello
for epoch in range(5):
    model.train() # Metto il modello in training
    running_loss = 0.0 # variabile che mi serve per salvarmi la loss
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        # se metto un breakpoint alla riga precedente e a console di debug scrivo: images.shape
        # ottengo: torch.Size([64, 1, 28, 28]) => questo mi dice che ho 64 batch di immagini che sono 28x28 per 1 canale (poichè MNIST è un DS in grayscale)
        # se a console scrivo: label.shape
        # ottengo: torch.Size([64]) => mi dice che ho 64 numeri
        # ad esempio: labels[0]
        # ottengo: tensor(7) => mi dice che il primo dato sarà di classe 7
        # Quindi, sposto tutto sulla GPU (se possibile)

        optimizer.zero_grad() # ripulisco l'ottimizzatore dai gradienti
        outputs = model(images) # faccio inferenza con la rete
        loss = criterion(outputs, labels) # calcolo la loss passandole l'output della rete e le label
        loss.backward() # calcolo i gradienti in modo automatico con autograd (predefinito in PyTorch)
        optimizer.step() # aggiorno i pesi

        # Nella versione più basic delle metriche, cumulo in una variabile le loss
        running_loss += loss.item()
    
    # Stampo la perdita di training, quindi divido la running_loss per la dimensione del dataloader 
    # (così ho la media delle loss che è stata calcolata su tutto il DS che è stata calcolata alla n-esima epoca)
    train_loss = running_loss / len(trainloader)
    print(f'Epoca {epoch + 1}, Perdita di training: {train_loss}') # Stampo la loss media di quella determinata epoca

    # Eseguo la fase di validazione
    # (la validazione posso farla: o ad ogni epoca di training; o ad ogni x epoche di training; posso non farla, anche se è meglio farla;
    # posso farla alla fine dell'addestramento.)
    model.eval() # Metto il modello in modalità di validazione
    # Variabili che uso per calcolarmi le loss e l'accuratezza
    running_loss = 0.0
    correct = 0
    total_val = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad(): # Disabilito il calcolo dei gradienti (rendo il tutto più leggero e veloce)
            outputs = model(images)
        loss = criterion(outputs, labels) # ugualmente, calcolo la loss

        ## Parte che riguarda l'accuratezza
        # Calcolo il numero di predizioni corrette
        _, predicted = torch.max(outputs, 1)
        total_val += labels.size(0)
        correct += (predicted == labels).sum().item() # le sommo tutte
    
    val_loss = running_loss / len(testloader)
    accuracy = 100 * correct / total_val # Le divido tutte per il totale per avere l'accuratezza totale
    print(f'Validazione - Perdita: {val_loss}, Accuratezza: {accuracy}%')

