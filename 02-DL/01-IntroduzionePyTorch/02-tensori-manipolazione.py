import torch

'''
Ogni tensore in PyTorch ha una serie di attributi che ne definiscono il comportamento e la struttura.
Alcuni sono:
- shape: indica il numero di elementi di ogni dimensione
- dtype: specifica il tipo di dato contenuto nel tensore (come float32, int64. Spesso si usa float32)
- device: indica dove il tensore è allocato (CPU, GPU o su TPU). Questo è molto importante per l'accelerazione del calcolo.
'''

tensor = torch.rand(3,4)
print( tensor )

print(f"Dimensione: {tensor.shape}")
print(f"Tipo di dato: {tensor.dtype}")
print(f"Dispositivo usato: {tensor.device}")
print()


##########################################################
############### MANIPOLAZIONE DEI TENSORI ################
##########################################################

'''
La manipolazione dei tensori è molto importante per la preparazione dei dati di addestramento dei modelli,
oppure per adattare l'output di un modello alle mie esigenze specifiche.

- SLICING: tensore[:,0]
           consente di selezionare porzioni specifiche di un tensore. Può essere utile per isolare determinati
           dati o features.
- RIDIMENSIONAMENTO: tensore.view( A, B ) 
                     modifica la forma di un tensore senza cambiare il contenuto. Spesso è importante fare questo per
                     conformare i dati agli input attesi di un modello.
- TRASPOSIZIONE: tensore.T() 
                 altera le dimensioni di un tensore, scambiando le righe con le colonne. Questo è utile per operazioni
                 matematiche o per adattare i formati dei dati.
'''

 
tensor2 = torch.arange(1, 10 ).view(3,3) # contiene i numeri da 1 a 9, e gli imposto la shape a 3x3
print("Tensore originale:\n", tensor2)


## SLICING
print("Seleziono la 1° colonna del tensore")
prima_colonna = tensor2[:, 0] # ottengo un nuovo tensore rappresentante la colonna
print("\nPrima colonna:\n", prima_colonna)

## RIDIMENSIONAMENTO
print("\nCambio la forma del tensore da 3x3 ad 1x9")
tensor_ridimensionato = tensor2.view(1, 9)
print("Tensore ridimensionato (1x9):\n", tensor_ridimensionato)

## TRASPOSIZIONE
tensor_trasporto = tensor2.t()
print("\nTensore trasposto:\n", tensor_trasporto)

'''
Altre operazioni importanti sono la concatenazione e la divisione dei tensori. Queste operazioni sono utili quando
lavoro con batch di dati, o quando voglio separare il Dataset in sottoinsieme. (Per "batch", nel ML ci si riferisce
ad un insieme di dati usati insieme in una singola iterazione dell'algoritmo di addestramento. In pratica, un batch
rappresenta una frazione del Dataset completo che viene usato per l'addestramento).

- CONCATENZAZIONE: unisce due o più tensori lungo una specifica dimensione. Quindi, mi consente di aggregare dati da
                    fonti diverse, o di combinare batch di dati.
- DIVISIONE: scompone un tensore in multiple parti di dimensioni predefiniti. Questo è utile per separare un grande
             Dataset in sottoinsiemi più gestibili

'''

#############################################################################
## Esempio per vedere l'applicazione della CONCATENZAZIONE e della DIVISIONE:
#############################################################################
'''Immagino di avere due batch di immagini, ciascuno contenente immagini rappresentate da tensori (2x3), dove 2
rappresenta il numero di immagini, e 3 le caratteristiche di ciascuna immagine.
Se voglio COMBINARE questi due batch in un unico set di dati per l'addestramento, posso usare la concatenazione.
Viceversa, se avessi un batch di 4 immagini e volessi dividerlo in 2 parti per validare il mio modello, dovrei usare
la DIVISIONE.'''

primo_tensore = torch.rand(2,3)
secondo_tensore = torch.rand(2,3)

## CONCATENAZIONE dei due tensori sulla dimensione 0 (cioè sul numero di immagini)
concatenazione = torch.cat( [primo_tensore, secondo_tensore], dim=0 )
print("\nTensori concatenati:\n", concatenazione )
print("Dimensione dopo la concatenazione: ", concatenazione.shape)

## DIVISIONE creo un tensore rappresentante un batch di 4 immagini
t1 = torch.rand(4, 4) #tensore di 4 immagini con 4 caratteristiche ciascuna
# divido il tensore in due parti, lungo la prima dimensione
split_tensor = torch.split(t1, 2) # dividi t1 in 2 parti
print("Le parti ottenute dalla divisione sono:")
for i,t in enumerate(split_tensor):
    print(f"Parte {i+1}:\n", t)

'''Ricapitolando:
- Nella parte di Concatenazione ho combinato primo_tensore e secondo_tensore in un unico tensore, lungo la prima dimensione (dim=0).
  Il risultato è stato un tensore (chiamato concatenazione) che rappresenta un batch di 4 immagini. Questo è utile quando si uniscono
  dati da batch diversi per creare un dataset più grande per l'addestramento.
- Nell'esempio di Divisione ho diviso un tensore in 2 parti uguali, ciascuna contenente 2 immagini e le relative caratteristiche. Questo
  mi permette di separare un batch di dati in sottoinsiemi più piccoli, come ad esempio per effettuare test o validazioni separate. 
'''