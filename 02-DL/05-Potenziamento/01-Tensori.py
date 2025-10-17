'''
Inizializzazione e Operazioni sui Tensori
Obiettivo: Crea un tensore 2D di dimensione 3×3 contenente numeri casuali, esegui operazioni di somma, prodotto e trasposizione su di esso
Inizializza un tensore di dimensione 3×3 con numeri casuali
Calcola la somma di tutti gli elementi del tensore
Calcola il prodotto tra il tensore originale e la sua trasposta
Verifica che il risultato sia simmetrico
'''

import torch

# Creo un tensore 2D di dimensione 3x3 con valori casuali
tensore = torch.rand(3, 3)

# Somma di tutti gli elementi
print('Somma degli elementi:\n', tensore.sum())

# Prodotto tra il tensore e la sua trasposta
tensore_trasposto = tensore.T
prodotto_tensore = torch.mm(tensore, tensore_trasposto)
print('Prodotto con la trasposta:\n', prodotto_tensore)

# Verifico se il risultato è simmetrico
is_simmetrico = torch.allclose(prodotto_tensore, prodotto_tensore.T)
print("Il risultato e' simmetrico: ", is_simmetrico)