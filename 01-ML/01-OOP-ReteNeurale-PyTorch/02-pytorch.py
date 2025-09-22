# Primo uso di PyTorch, per vedere il suo approccio OOP. Inoltre,
# viene mostrato come creare un modello, ereditando dalla classe nn.Module

# (1) Il modulo torch.nn, che contiene le classi e le funzioni per
#     costruire reti neurali. La uso per definire e istanziare i miei layer,
#     come quelli convoluzionali ad esempio.
import torch.nn as nn # 

# (2) Il modulo functional fornisce funzioni per operazioni come Es): funzione di
#     attivazione; il pooling; etc.
#     Così facendo posso applicare queste operazioni ai TENSORI.
import torch.nn.functional as F


# (3) La classe estende nn.Module, che è la classe base per tutti i modelli neurali in PyTorch.
class Mio_Modello(nn.Module):
    def __init__(self):

        # (4) Inizializzo la classe chiamando il costruttore della superclasse nn.Module.
        #     (Questo passo è necessario per inizializzare correttamente la rete)
        super(Mio_Modello, self).__init__()

        # (5) Con le seguenti istruzioni definisco i layer del mio modello di Rete Neurale
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10) 

    # (6) Definisco il metodo forward 
    def forward(self, x):
        # Questo metodo definisce come i dati vengono trasformati al modello:
        # (funzione di attivazione, pooling, etc.). 
        # Serve ad implementare i vari passaggi del forward
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


        

