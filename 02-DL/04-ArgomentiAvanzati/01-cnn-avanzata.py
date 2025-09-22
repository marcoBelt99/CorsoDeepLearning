import torch
import torch.nn as nn # modulo che contiene gli strumenti per costruire le Reti Neurali
import torch.nn.functional as F # functional contiene le funzioni di attivazione e altre utilità


class AdvancedCNN(nn.Module): # AdvancedCNN estende la classe nn.Module
    def __init__(self):
        '''
        Metodo di inizializzazione della classe
        '''
        # Inizializzo la parte della classe che proviene da nn.Module
        super(AdvancedCNN, self).__init__()

        ## Definisco gli strati della rete
        #   Definisco conv1: uno strato di convoluzione che prende immagini con 3 canali (tipicamente RGB),
        #   e produce 64 canali in uscita. Il kernel_size=3 e il padding=1 determinano rispettivamente
        #   la dimensione del filtro e l'aggiunta di bordi.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        #   Definisco bn1: uno strato di batch normalization. Gli strati di batch normalization aiutano a 
        #   stabilizzare e accelerare la convergenza durante l'addestramento.
        self.bn1 = nn.BatchNorm2d(64)
        #   Definisco conv2: strato convoluzionale che prende i 64 canali in uscita da conv1 e produce 128 canali
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
       #    Definisco bn2: un altro strato di batch normalization
        self.bn2 = nn.BatchNorm2d(128)
        #   Definisco fc: un ultimo strato completamente connesso, che mappa le caratteristiche estratte dai livelli convoluzionali
        #   a 10 classi di output (magari per un problema di classificazione con 10 classi)
        self.fc = nn.Linear(128 * 7 * 7, 10)

    def forward(self, x):
        '''
        Metodo che definisce come i dati passano attraverso la rete (forward pass).
        '''
        ## Questo applica il primo strato convoluzionale, seguito dalla normalizzazione batch e dalla funzione di attivazione ReLU
        x = F.relu( self.bn1( self.conv1( x ) ) )

        ## Applico il pooling per ridurre la dimensione spaziale (larghezza e altezza)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        ## Ripeto il processo per i secondi strati convoluzionali e di batch normalization
        x = F.relu( self.bn2( self.conv2( x ) ) )
        x = F.max_pool2d( x, kernel_size=2, stride=2 )

        ## flatten: questo appiattisce tutte le dimensioni tranne la prima (batch_size), preparando i dati per
        #  lo strato completamente connesso
        x = torch.flatten(x, 1)

        ## Passo i dati appiattiti attraverso lo strato completamente connesso per ottenere i punteggi finali della classi
        x = self.fc( x )
