class ReteNeurale:
    # (1) Costruttore: viene chiamato quando viene creata un'istanza della classe.
    #     E' qui che inizializzo gli strati della rete. 
    def __init__(self, input_size, hidden_size, output_size):
        '''
        input_size: dimensione dell'input. Quante entrate deve gestire la rete. Es) per
                    riconoscere una lettera, l'input_size potrebbe essere il numero di px
                    dell'immagine che rappresenta una lettera.
        hidden_size: dimensione dello strato nascosto. I neuroni di questo strato elaborano
                     l'informazione.
        output_size: dimensione dell'output, ossia quante classi o risultati la mia rete può
                     produrre. Es) nell'esempio, potrebbe essere il numero di lettere dell'alfabeto
        '''

        # (2) Definisco e creo i layer della rete. La classe Layer è una sorta di contenitore tra neuroni.
        self.strato1 = Layer(input_size, hidden_size)
        # gli strati (layer) collegano l'input con l'output attraverso uno strato nascosto
        self.strato2 = Layer(hidden_size, output_size)

    
    # (3) Definizione del metodo forward: definisce come i dati attraversano la rete (il passaggio in avanti)
    #     Questo è fondamentale per calcolare l'output della rete basato sull'input.
    def forward(self, x):
        '''
        E' in questo metodo che i dati vengono effettivamente processati.
        Il parametro x è il dato (o l'insieme di dati) che voglio che la mia rete elabori.
        '''
        # Questo metodo simula il passaggio dei dati attraverso gli strati della rete, applicando
        # una determinata funzione di attivazione, dopo il primo strato. La funzione di attivazione ha
        # il compito di decidere se un neurone si deve accendere o meno, basandosi sui dati che riceve
        x = self.strato1(x)
        x = activation_function(x)
        x = self.strato2(x)
        return x
    
    # (4) Scrittura del metodo di training. Qui viene implementata la logica per addestrare la rete sui dati.
    #     Qui possono essere incluse la definizione di: una funzione di perdita; dell'ottimizzazione. La specifica
    #     di come i pesi della rete vengono aggiornati in base ai dati di addestramento; etc. 
    def addestrare(self, dati_di_addestramento):
        '''
        In questo metodo dovrebbero esserci: aggiustamento di pesi; calcolo di perdita; backpropagation; etc.
        '''
        pass

    # (5) Scrittura del metodo per fare previsioni. Qui viene implementata la logica per effettuare previsioni su nuovi dati.
    def predire(self, dati_di_test):
        '''
        Questo metodo mi permette di usare la rete per fare previsioni su dati mai visti prima.
        '''
        predizioni = []

        # Tramite il ciclo, elaboro ogni elemento dei dati di test che passo,
        # usando il metodo forward(), e colleziono i risultati in una lista di predizioni.
        for dato in dati_di_test:
            predizione = self.forward(dato)
            predizioni.append(predizione)

