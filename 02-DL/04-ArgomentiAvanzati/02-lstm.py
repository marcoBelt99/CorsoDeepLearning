import torch
import torch.nn as nn

class SentimentLSTM(nn.Module): # estendo la classe base nn.Module, che mi obbliga a definire almeno 2 metodi (init e forward)
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        '''
        I parametri passati sono:
        - vocab_size: numero di parole del vocabolario
        - embedding_dim: dimensione degli embedding, cioè quanti numeri saranno usati per rappresentare ogni parola.
        - hidden_dim: dimensione dello strato nascosto dalla LSTM
        - output_dim: dimensione dell'output finale del modello, che tipicamente è il numero delle classe 
        '''
        # Richiamo il costruttore della classe base, per poter inizializzare correttamente la parte di RN del mio progetto
        super(SentimentLSTM, self).__init__()

        ### Componenti della rete
        
        ## Creo una matrice di embedding in cui ogni riga rappresenta un embedding di una parola
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        ## Definisco uno strato LSTM che prende gli input dalla dimensione degli embedding e trasmette
        #  output alla dimensione dello strato nascosto 
        self.lstm = nn.LSTM( embedding_dim, hidden_dim )

        ## Definisco lo strato lineare, il quale mappa lo strato nascosto finale alla dimensione dell'output
        #  ad esempio, il numero di classi per la classificazione.
        self.fc = nn.Linear( hidden_dim, output_dim )

    def forward(self, text):
        '''
        Metodo che definisce come i dati passano attraverso la rete
        ''' 
        ## Converto gli indici delle parole in vettori di embedding
        embedded = self.embedding( text )
        
        ## Passo gli embedding attraverso la LSTM
        output, (hidden, _) = self.embedding #output contiene tutti gli outpt, hidden contiene lo strato nascosto finale

        ##  Rimuovo una dimensione unitaria dall'output per prepararlo per il layer finale.
        ##  questo è necessario solo se la LSTM è ad un solo strato!
        hidden = hidden.squeeze(0)

        ## Passo lo strato nascosto attraverso il layer lineare per ottenere l'output finale.
        out = self.fc( hidden )

        return out
    

'''
Quindi: questo modello prende come input una sequenza di indici di parole che rappresentano una frase o ad esempio
        un documento, e restituisce un output che può essere usato per una classificazione o per altre attività legate
        all'analisi del sentiment.
        Si inizia con un layer di embedding, seguito da un layer LSTM, e infine di un layer completamente connesso
        che produce l'output finale.
    La bellezza di PyTorch è che posso assemblare questi componenti come delle costruzioni lego, costruendo modelli
    complessi con una relativa facilità.
'''