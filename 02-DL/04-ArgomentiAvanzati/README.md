### 01-cnn-avanzata.py

Creo una classe `AdvancedCNN` che definisce appunto una Rete Neurale Convoluzionale per lavorare con immagini.
In particolare, al suo interno definisco una rete neurale che può essere usata per classificare immaini basate su 10
possibili classi. Utilizza dei layer convoluzionali per l'estrazione delle features, seguiti da layer di pooling per 
la riduzione dimensionale, e infine un layer completamente connesso per la classificazione finale.

### 02-lstm.py
PyTorch è un grande perchè mi semplifica notevolemente l'implementazione di LSTM e GRU, tramite dei moduli
pre-costruiti.
In questo esempio costruisco un semplice modello LSTM per classificare il sentiment di recensioni di film, imparando
passo dopo passo come preparare questo modello.

### 03-gru.py
Codice che definisce un modello GRU base che può essere addestrato per svolgere dei compiti come la classificazione del testo.
L'input del modello sarà una sequenza di vettori (tipicamente embedding di parole), e l'output sarà un vettore di classificazione
per l'intera sequenza.