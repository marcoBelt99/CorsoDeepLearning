import torch

#####################################
############### BASI ################
#####################################


# Vettore (1D) che contiene 2 elementi
print('--- Tensore 1D (vettore) ---')
vector = torch.tensor( [3,2] )

# Per contare la dimensione di un tensore posso contare il numero di parentesi quadre aperte.
# Oppure posso usare
print( f'Numero di dimensioni: {vector.ndim}' )

# Forma di un tensore: mi dice come sono organizzati gli elementi al suo interno
# nel caso di vector, la forma è torch.size[2] e significa che contiene 2 elementi.
print( f'Shape del vettore: {vector.shape}')

'''
NOTA BENE:
- ndim:  mi dice quante dimensioni ha il tensore
- shape: mi dice la lunghezza di ciascuna dimensione del tensore
'''


# Matrice: aggiunge un'altra dimensione al vettore
# (la matrice è 2D)
print('\n--- Tensore 2D (matrice) ---')
MATRIX = torch.tensor( [[7,8], [9,10]] ) # visto che è 2D, nota bene che ho 2 parentesi!
print( MATRIX )


# Tensore:
print('\n--- Tensore generico ---')
TENSOR = torch.tensor( [[[1,2,3], [3,6,9], [2,4,5]]] )
print(f'Numero di dimensioni:  + {TENSOR.ndim}')
print(f'Shape:  + {TENSOR.shape}')
# In questo caso la shape è:
#> 3
#> toch.Size([1, 3, 3])
# 1 indica che c'è 1 sola riga nel tensore; 3 (al centro) indica che ci sono 3 colonne in ogni riga; 
# 3 (a destra) indica che ci sono 3 valori all'interno di ogni colonna

# Ad esempio, quel tensore potrebbe rappresentare i numeri di vendita di vari prodotti in diversi giorni.


#####################################
########### ALTRI METODI ############
#####################################

# PyTorch offre anche altri metodi per creare dei tensori con valori predefiniti, che sono utili per vari scopi,
# come ad esempio l'inizializzazione dei pesi in una rete neurale.

# Creo un tensore di zeri
zeros = torch.zeros( (2,3) )

# Tensore con tutti 1
ones = torch.ones( (3, 3) )

# Tensore con tutti valori casuali
random = torch.rand( (2,2) )

'''
Come vedo, i tensori di 0 e di 1 sono usati spesso per inizializzare pesi e per creare delle maschere.
Invece, i tensori con valori casuali sono molto importanti per iniziare l'addestramento dei modelli con 
pesi casuali.
'''

# Per usare un range di numeri da 1 a 10 uso:
zero_to_cento = torch.arange(start=0, end=10, step=1) # con lo step