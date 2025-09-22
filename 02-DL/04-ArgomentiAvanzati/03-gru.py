import torch
import torch.nn as nn

class GRUModel(nn.Module):
    '''
    Definisco la classe GRUModel che eridata da nn.Module (che è la classe base per tutti i moduli di RN in PyTorch)
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        - input_dim:    è la dimensione dell'input per ogni timestep
        - hidden_dim:   è la dimensione dello strato nascosto della GRU
        - output_dim:   è la dimensione dello strato di output desiderato.
        '''
        # Richiamo il costruttore della classe base per fare in modo che GRUModel si inizializzi come un modulo PyTorch
        super( GRUModel, self ).__init__()
        
        ## Memorizzo la dimensione dello strato nascosto passata al costruttore
        self.hidden_dim = hidden_dim 
        
        #### Definisco i layers della rete 
        ## Creo un layer GRU
        self.gru = nn.GRU( input_dim, hidden_dim, batch_first=True  ) # batch_first=True indica che l'input e l'output tensors saranno con 
        # la dimensione del batch come prima dimensione

        ## Creo lo strato lineare (o FC)
        self.fc = nn.Linear( hidden_dim, output_dim )
    
    def forward(self, x):
        '''
        Definisco il passaggio in avanti del modello.
        - x è l'input al modello.
        '''

        ## Inizializzo lo strato nascosto per la GRU a 0.
        #   1 indica il numero di layer nella GRU; 
        #   x.size(0) è la dimensione del batch; 
        #   self.hidden_dim è la dimensione dello strato nascosto
        #   .to(x.device) mi assicuro che lo strato nascosto sia sullo stesso dispositivo dell'input (CPU o GPU)
        h0 = torch.zeros( 1, x.size(0), self.hidden_dim ).to( x.device )

        ## Eseguo il layer gru.
        #  out contiene tutti gli output della gru per ogni time-step;
        #  hn è l'ultimo strato nascosto
        out, hn = self.gru( x, h0 )

        ## Applico il layer lineare all'ultimo output della sequenza
        out = self.fc( out[:, -1, :] )

        # Il risultato è l'output finale del modello
        return out
    

#### Istanzio e visualizzo il modello

# Specifico le dimensioni di input, hidden e output
input_dim = 10
hidden_dim = 50
output_dim = 2

# Inizializzo il modello
model = GRUModel( input_dim, hidden_dim, output_dim )

# Stampo la struttura del modello per verificare che sia configurato come desiderato.
print( model )